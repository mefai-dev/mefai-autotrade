"""
Configuration Manager - YAML-based configuration for Mefai Autotrade.
Provides config loading, validation, hot-reload, secrets management,
environment variable overrides, per-strategy and per-exchange configuration.
"""

import base64
import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("mefai.autotrade.config_manager")


# ---------------------------------------------------------------------------
# YAML loading (lightweight, no external dependency beyond stdlib)
# ---------------------------------------------------------------------------

def _load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load a YAML file. Uses PyYAML if available, falls back to a basic
    key-value parser for simple configs.
    """
    try:
        import yaml
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return _basic_yaml_parse(filepath)


def _basic_yaml_parse(filepath: str) -> Dict[str, Any]:
    """
    Minimal YAML parser for flat and one-level-nested configs.
    Handles: key: value, key: "string", nested keys with 2-space indent.
    Not a full YAML parser - install PyYAML for complex configs.
    """
    result: Dict[str, Any] = {}
    current_section = ""

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.startswith("#"):
                continue

            # Count leading spaces
            indent = len(line) - len(line.lstrip())

            if ":" not in stripped:
                continue

            parts = stripped.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""

            if indent == 0:
                if value == "" or value == "":
                    current_section = key
                    result[current_section] = {}
                else:
                    result[key] = _parse_value(value)
                    current_section = ""
            elif indent >= 2 and current_section:
                if isinstance(result.get(current_section), dict):
                    result[current_section][key] = _parse_value(value)

    return result


def _parse_value(value: str) -> Any:
    """Parse a YAML value string into a Python type."""
    if not value:
        return ""

    # Remove quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    # Boolean
    lower = value.lower()
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False
    if lower in ("null", "none", "~"):
        return None

    # Numbers
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # Lists (simple inline format)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        if not inner.strip():
            return []
        return [_parse_value(v.strip()) for v in inner.split(",")]

    return value


def _dump_yaml(data: Dict[str, Any], filepath: str) -> None:
    """Write config data to a YAML file."""
    try:
        import yaml
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        _basic_yaml_dump(data, filepath)


def _basic_yaml_dump(data: Dict[str, Any], filepath: str) -> None:
    """Basic YAML dumper for simple configs."""
    lines: List[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k}: {_format_value(v)}")
        else:
            lines.append(f"{key}: {_format_value(value)}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def _format_value(value: Any) -> str:
    """Format a Python value as a YAML string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if any(c in value for c in ":#{}[]|>!&*?,"):
            return f'"{value}"'
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    return str(value)


# ---------------------------------------------------------------------------
# Config schema and validation
# ---------------------------------------------------------------------------

class ConfigFieldType:
    """Configuration field type constants."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


@dataclass
class ConfigField:
    """Schema for a single configuration field."""
    name: str = ""
    field_type: str = ConfigFieldType.STRING
    required: bool = False
    default: Any = None
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    env_var: Optional[str] = None
    secret: bool = False


@dataclass
class ConfigSchema:
    """Schema definition for a configuration section."""
    name: str = ""
    description: str = ""
    fields: List[ConfigField] = field(default_factory=list)

    def get_field(self, name: str) -> Optional[ConfigField]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def get_defaults(self) -> Dict[str, Any]:
        return {f.name: f.default for f in self.fields if f.default is not None}


class ConfigValidator:
    """Validate configuration values against a schema."""

    def __init__(self):
        self._schemas: Dict[str, ConfigSchema] = {}

    def register_schema(self, schema: ConfigSchema) -> None:
        self._schemas[schema.name] = schema

    def validate(
        self, section: str, config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate a config section. Returns (is_valid, list_of_errors)."""
        schema = self._schemas.get(section)
        if schema is None:
            return True, []  # No schema = no validation

        errors: List[str] = []

        for fld in schema.fields:
            value = config.get(fld.name)

            if value is None:
                if fld.required:
                    errors.append(f"{section}.{fld.name}: required field is missing")
                continue

            # Type checking
            type_valid = True
            if fld.field_type == ConfigFieldType.STRING:
                type_valid = isinstance(value, str)
            elif fld.field_type == ConfigFieldType.INTEGER:
                type_valid = isinstance(value, int) and not isinstance(value, bool)
            elif fld.field_type == ConfigFieldType.FLOAT:
                type_valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif fld.field_type == ConfigFieldType.BOOLEAN:
                type_valid = isinstance(value, bool)
            elif fld.field_type == ConfigFieldType.LIST:
                type_valid = isinstance(value, list)
            elif fld.field_type == ConfigFieldType.DICT:
                type_valid = isinstance(value, dict)

            if not type_valid:
                errors.append(
                    f"{section}.{fld.name}: expected {fld.field_type}, "
                    f"got {type(value).__name__}"
                )
                continue

            # Range checking
            if fld.min_value is not None and isinstance(value, (int, float)):
                if value < fld.min_value:
                    errors.append(
                        f"{section}.{fld.name}: value {value} below minimum {fld.min_value}"
                    )

            if fld.max_value is not None and isinstance(value, (int, float)):
                if value > fld.max_value:
                    errors.append(
                        f"{section}.{fld.name}: value {value} above maximum {fld.max_value}"
                    )

            # Choices checking
            if fld.choices is not None and value not in fld.choices:
                errors.append(
                    f"{section}.{fld.name}: value '{value}' not in allowed choices {fld.choices}"
                )

        return len(errors) == 0, errors

    def validate_all(
        self, config: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate all sections in a config dict."""
        all_errors: Dict[str, List[str]] = {}
        all_valid = True

        for section_name, schema in self._schemas.items():
            section_config = config.get(section_name, {})
            if not isinstance(section_config, dict):
                section_config = {}
            valid, errors = self.validate(section_name, section_config)
            if not valid:
                all_valid = False
                all_errors[section_name] = errors

        return all_valid, all_errors


# ---------------------------------------------------------------------------
# Secrets management
# ---------------------------------------------------------------------------

class SecretManager:
    """
    Manage sensitive configuration values.
    Provides encryption at rest using a key derived from a passphrase.
    """

    def __init__(self, key: Optional[str] = None):
        self._key = key or os.environ.get("MEFAI_CONFIG_KEY", "mefai-autotrade-default")
        self._secrets: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _derive_key(self) -> bytes:
        """Derive a 32-byte key from the passphrase."""
        return hashlib.sha256(self._key.encode()).digest()

    def _xor_encrypt(self, data: str) -> str:
        """Simple XOR encryption with the derived key. Not cryptographically strong -
        use a proper encryption library (Fernet) for production secrets."""
        key = self._derive_key()
        data_bytes = data.encode()
        encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data_bytes))
        return base64.b64encode(encrypted).decode()

    def _xor_decrypt(self, data: str) -> str:
        """Decrypt XOR-encrypted data."""
        key = self._derive_key()
        encrypted = base64.b64decode(data.encode())
        decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(encrypted))
        return decrypted.decode()

    def set_secret(self, name: str, value: str) -> None:
        """Store an encrypted secret."""
        with self._lock:
            self._secrets[name] = self._xor_encrypt(value)

    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve and decrypt a secret."""
        with self._lock:
            encrypted = self._secrets.get(name)
        if encrypted is None:
            return None
        return self._xor_decrypt(encrypted)

    def has_secret(self, name: str) -> bool:
        return name in self._secrets

    def delete_secret(self, name: str) -> bool:
        with self._lock:
            return self._secrets.pop(name, None) is not None

    def list_secret_names(self) -> List[str]:
        return list(self._secrets.keys())

    def save_to_file(self, filepath: str) -> None:
        """Save encrypted secrets to a JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self._secrets, f, indent=2)
        # Set restrictive permissions
        os.chmod(filepath, 0o600)

    def load_from_file(self, filepath: str) -> None:
        """Load encrypted secrets from a JSON file."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r") as f:
            self._secrets = json.load(f)

    def clear(self) -> None:
        with self._lock:
            self._secrets.clear()


# ---------------------------------------------------------------------------
# Config file watcher
# ---------------------------------------------------------------------------

class ConfigWatcher:
    """Watch config files for changes and trigger reload."""

    def __init__(self, interval: float = 5.0):
        self._interval = interval
        self._files: Dict[str, float] = {}  # filepath -> last modified time
        self._callbacks: List[Callable[[str], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def watch(self, filepath: str) -> None:
        """Add a file to watch list."""
        with self._lock:
            try:
                mtime = os.path.getmtime(filepath)
                self._files[filepath] = mtime
            except OSError:
                self._files[filepath] = 0.0

    def unwatch(self, filepath: str) -> None:
        with self._lock:
            self._files.pop(filepath, None)

    def on_change(self, callback: Callable[[str], None]) -> None:
        """Register a callback for file changes. Callback receives the filepath."""
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="config-watcher"
        )
        self._thread.start()
        logger.info("Config watcher started (interval: %ss)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    def _watch_loop(self) -> None:
        while self._running:
            try:
                self._check_files()
            except Exception as exc:
                logger.error("Config watcher error: %s", exc)
            time.sleep(self._interval)

    def _check_files(self) -> None:
        with self._lock:
            files = dict(self._files)

        for filepath, last_mtime in files.items():
            try:
                current_mtime = os.path.getmtime(filepath)
                if current_mtime > last_mtime:
                    with self._lock:
                        self._files[filepath] = current_mtime
                    logger.info("Config file changed: %s", filepath)
                    for cb in self._callbacks:
                        try:
                            cb(filepath)
                        except Exception as exc:
                            logger.error("Config change callback error: %s", exc)
            except OSError:
                pass

    @property
    def watched_files(self) -> List[str]:
        with self._lock:
            return list(self._files.keys())


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "name": "Mefai Autotrade",
        "mode": "paper",  # paper, live
        "log_level": "INFO",
        "data_dir": "data",
        "timezone": "UTC",
    },
    "database": {
        "type": "sqlite",
        "path": "autotrade.db",
        "pool_size": 5,
        "journal_mode": "WAL",
        "retention_days": 365,
    },
    "risk": {
        "max_position_size_pct": 5.0,
        "max_total_exposure_pct": 30.0,
        "max_daily_loss_pct": 3.0,
        "max_drawdown_pct": 15.0,
        "max_leverage": 10.0,
        "max_open_positions": 10,
        "correlation_limit": 0.7,
        "emergency_stop": True,
    },
    "execution": {
        "default_order_type": "MARKET",
        "slippage_tolerance_pct": 0.1,
        "retry_attempts": 3,
        "retry_delay_seconds": 1.0,
        "rate_limit_per_second": 5,
        "confirm_orders": False,
    },
    "market_data": {
        "stale_threshold_seconds": 10.0,
        "price_deviation_pct": 5.0,
        "tick_buffer_size": 50000,
        "candle_buffer_size": 5000,
        "persist_interval_seconds": 60,
    },
    "strategies": {},
    "exchanges": {},
}


# ---------------------------------------------------------------------------
# Main config manager
# ---------------------------------------------------------------------------

class ConfigManager:
    """
    Central configuration management for Mefai Autotrade.
    Supports YAML files, environment variable overrides, validation,
    hot-reload, secrets, and per-strategy/exchange configs.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        secrets_path: Optional[str] = None,
        auto_reload: bool = False,
        env_prefix: str = "MEFAI_",
    ):
        self._config_path = config_path
        self._env_prefix = env_prefix
        self._config: Dict[str, Any] = deepcopy(DEFAULT_CONFIG)
        self._validator = ConfigValidator()
        self._secret_manager = SecretManager()
        self._watcher = ConfigWatcher()
        self._lock = threading.RLock()
        self._change_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Load config file if provided
        if config_path and os.path.exists(config_path):
            self.load(config_path)

        # Load secrets if provided
        if secrets_path and os.path.exists(secrets_path):
            self._secret_manager.load_from_file(secrets_path)

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Setup hot-reload
        if auto_reload and config_path:
            self._watcher.watch(config_path)
            self._watcher.on_change(self._on_file_change)
            self._watcher.start()

        logger.info("ConfigManager initialized")

    # --- Loading ---

    def load(self, filepath: str) -> None:
        """Load configuration from a YAML file."""
        data = _load_yaml(filepath)
        with self._lock:
            self._merge_config(self._config, data)
            self._config_path = filepath
        logger.info("Loaded config from %s", filepath)

    def reload(self) -> None:
        """Reload configuration from the current file."""
        if self._config_path and os.path.exists(self._config_path):
            old_config = deepcopy(self._config)
            self._config = deepcopy(DEFAULT_CONFIG)
            self.load(self._config_path)
            self._apply_env_overrides()

            # Notify change callbacks
            for cb in self._change_callbacks:
                try:
                    cb(self._config)
                except Exception as exc:
                    logger.error("Config change callback error: %s", exc)

            logger.info("Config reloaded from %s", self._config_path)

    def _on_file_change(self, filepath: str) -> None:
        """Called by file watcher when config file changes."""
        logger.info("Config file changed, reloading: %s", filepath)
        self.reload()

    def on_change(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for configuration changes."""
        self._change_callbacks.append(callback)

    # --- Getters ---

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value using dot notation.
        Example: get("risk.max_leverage") returns config["risk"]["max_leverage"]
        """
        with self._lock:
            return self._deep_get(self._config, key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a full configuration section."""
        with self._lock:
            value = self._config.get(section, {})
            return deepcopy(value) if isinstance(value, dict) else {}

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        with self._lock:
            strategies = self._config.get("strategies", {})
            return deepcopy(strategies.get(strategy_name, {}))

    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """Get configuration for a specific exchange."""
        with self._lock:
            exchanges = self._config.get("exchanges", {})
            return deepcopy(exchanges.get(exchange_name, {}))

    def get_all(self) -> Dict[str, Any]:
        """Get the full configuration (deep copy)."""
        with self._lock:
            return deepcopy(self._config)

    # --- Setters ---

    def set(self, key: str, value: Any) -> None:
        """
        Set a config value using dot notation.
        Example: set("risk.max_leverage", 5.0)
        """
        with self._lock:
            self._deep_set(self._config, key, value)

    def set_strategy_config(self, strategy_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific strategy."""
        with self._lock:
            if "strategies" not in self._config:
                self._config["strategies"] = {}
            self._config["strategies"][strategy_name] = config

    def set_exchange_config(self, exchange_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific exchange."""
        with self._lock:
            if "exchanges" not in self._config:
                self._config["exchanges"] = {}
            self._config["exchanges"][exchange_name] = config

    # --- Validation ---

    def register_schema(self, schema: ConfigSchema) -> None:
        """Register a validation schema for a config section."""
        self._validator.register_schema(schema)

    def validate(self, section: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Validate configuration.
        If section is provided, validates only that section.
        Otherwise validates all sections with registered schemas.
        """
        with self._lock:
            if section:
                section_config = self._config.get(section, {})
                if not isinstance(section_config, dict):
                    section_config = {}
                return self._validator.validate(section, section_config)
            return self._validator.validate_all(self._config)

    # --- Secrets ---

    @property
    def secrets(self) -> SecretManager:
        return self._secret_manager

    def get_secret(self, name: str) -> Optional[str]:
        return self._secret_manager.get_secret(name)

    def set_secret(self, name: str, value: str) -> None:
        self._secret_manager.set_secret(name, value)

    # --- Environment overrides ---

    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides.
        Format: MEFAI_SECTION_KEY=value (e.g., MEFAI_RISK_MAX_LEVERAGE=5.0)
        """
        prefix = self._env_prefix.upper()

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Convert ENV_KEY_NAME to config path: section.key_name
            parts = env_key[len(prefix):].lower().split("_", 1)
            if len(parts) < 2:
                continue

            section = parts[0]
            key = parts[1]

            # Try to parse the value to the right type
            parsed = _parse_value(env_value)

            with self._lock:
                if section in self._config and isinstance(self._config[section], dict):
                    self._config[section][key] = parsed
                    logger.debug("Applied env override: %s.%s = %s", section, key, parsed)

    # --- Config diff ---

    def diff(self, other: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare current config against defaults or another config.
        Returns dict of differences.
        """
        reference = other or DEFAULT_CONFIG
        differences: Dict[str, Any] = {}

        with self._lock:
            self._find_diffs(self._config, reference, "", differences)

        return differences

    def _find_diffs(
        self,
        current: Dict[str, Any],
        reference: Dict[str, Any],
        prefix: str,
        diffs: Dict[str, Any],
    ) -> None:
        all_keys = set(list(current.keys()) + list(reference.keys()))
        for key in sorted(all_keys):
            full_key = f"{prefix}.{key}" if prefix else key
            cur_val = current.get(key)
            ref_val = reference.get(key)

            if isinstance(cur_val, dict) and isinstance(ref_val, dict):
                self._find_diffs(cur_val, ref_val, full_key, diffs)
            elif cur_val != ref_val:
                diffs[full_key] = {
                    "current": cur_val,
                    "default": ref_val,
                }

    # --- Export / Import ---

    def save(self, filepath: Optional[str] = None) -> None:
        """Save current configuration to a YAML file."""
        path = filepath or self._config_path
        if not path:
            raise ValueError("No file path specified")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            _dump_yaml(self._config, path)
        logger.info("Config saved to %s", path)

    def export_json(self) -> str:
        """Export configuration as JSON string."""
        with self._lock:
            return json.dumps(self._config, indent=2, default=str)

    def import_json(self, json_str: str) -> None:
        """Import configuration from a JSON string."""
        data = json.loads(json_str)
        with self._lock:
            self._merge_config(self._config, data)

    # --- Internal helpers ---

    @staticmethod
    def _deep_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get nested dict value using dot notation."""
        keys = key.split(".")
        current = d
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k)
                if current is None:
                    return default
            else:
                return default
        return current

    @staticmethod
    def _deep_set(d: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested dict value using dot notation."""
        keys = key.split(".")
        current = d
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    @staticmethod
    def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge override into base."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                ConfigManager._merge_config(base[key], value)
            else:
                base[key] = value

    # --- Lifecycle ---

    def close(self) -> None:
        """Stop watcher and release resources."""
        self._watcher.stop()
        logger.info("ConfigManager closed")

    def __enter__(self) -> "ConfigManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            sections = list(self._config.keys())
            strategy_count = len(self._config.get("strategies", {}))
            exchange_count = len(self._config.get("exchanges", {}))

        return {
            "config_path": self._config_path,
            "sections": sections,
            "strategy_configs": strategy_count,
            "exchange_configs": exchange_count,
            "secrets_count": len(self._secret_manager.list_secret_names()),
            "watched_files": self._watcher.watched_files,
            "env_prefix": self._env_prefix,
        }
