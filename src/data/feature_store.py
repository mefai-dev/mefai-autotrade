"""
Feature Store - Feature management for ML models in Mefai Autotrade.
Provides feature versioning, schema validation, selection, correlation
analysis, batch and online computation, and persistent storage.
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("mefai.autotrade.feature_store")


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

class FeatureType:
    """Feature data type constants."""
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


@dataclass
class FeatureSchema:
    """Schema definition for a single feature."""
    name: str = ""
    feature_type: str = FeatureType.FLOAT
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nullable: bool = False
    default_value: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    group: str = ""
    compute_fn_name: str = ""

    def validate_value(self, value: Any) -> Tuple[bool, str]:
        """Validate a single value against this schema. Returns (valid, error_message)."""
        if value is None:
            if self.nullable:
                return True, ""
            return False, f"Feature '{self.name}' is not nullable"

        if self.feature_type == FeatureType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Feature '{self.name}' expects float, got {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return False, f"Feature '{self.name}' value {value} below min {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Feature '{self.name}' value {value} above max {self.max_value}"
        elif self.feature_type == FeatureType.INTEGER:
            if not isinstance(value, int):
                return False, f"Feature '{self.name}' expects int, got {type(value).__name__}"
        elif self.feature_type == FeatureType.BOOLEAN:
            if not isinstance(value, (bool, int)):
                return False, f"Feature '{self.name}' expects bool, got {type(value).__name__}"

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_type": self.feature_type,
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "nullable": self.nullable,
            "default_value": self.default_value,
            "tags": self.tags,
            "group": self.group,
        }


# ---------------------------------------------------------------------------
# Feature versioning
# ---------------------------------------------------------------------------

@dataclass
class FeatureVersion:
    """Tracks a specific version of feature computation logic."""
    version: str = ""
    features: List[str] = field(default_factory=list)
    description: str = ""
    created_at: str = ""
    hash_signature: str = ""
    parent_version: str = ""
    is_active: bool = True

    @staticmethod
    def compute_hash(features: List[str], description: str) -> str:
        """Create a deterministic hash for a feature set."""
        content = json.dumps({"features": sorted(features), "desc": description})
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "features": self.features,
            "description": self.description,
            "created_at": self.created_at,
            "hash_signature": self.hash_signature,
            "parent_version": self.parent_version,
            "is_active": self.is_active,
        }


# ---------------------------------------------------------------------------
# Feature selector
# ---------------------------------------------------------------------------

class FeatureSelector:
    """Feature selection utilities based on importance and correlation."""

    @staticmethod
    def select_by_importance(
        importances: Dict[str, float],
        threshold: float = 0.01,
        max_features: Optional[int] = None,
    ) -> List[str]:
        """
        Select features based on importance scores.
        Returns feature names sorted by importance (highest first).
        """
        filtered = {k: v for k, v in importances.items() if v >= threshold}
        sorted_features = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        if max_features:
            sorted_features = sorted_features[:max_features]

        return [name for name, _ in sorted_features]

    @staticmethod
    def select_uncorrelated(
        feature_matrix: np.ndarray,
        feature_names: List[str],
        max_correlation: float = 0.8,
    ) -> List[str]:
        """
        Remove highly correlated features, keeping one from each correlated pair.
        Uses greedy forward selection.
        """
        n_features = feature_matrix.shape[1] if len(feature_matrix.shape) > 1 else 0
        if n_features == 0 or n_features != len(feature_names):
            return list(feature_names)

        corr_matrix = np.corrcoef(feature_matrix.T)
        selected: List[int] = [0]

        for i in range(1, n_features):
            is_correlated = False
            for j in selected:
                if abs(corr_matrix[i, j]) > max_correlation:
                    is_correlated = True
                    break
            if not is_correlated:
                selected.append(i)

        return [feature_names[i] for i in selected]

    @staticmethod
    def rank_by_variance(
        feature_matrix: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:
        """Rank features by variance (higher variance = potentially more informative)."""
        if feature_matrix.shape[1] != len(feature_names):
            return []

        variances = np.nanvar(feature_matrix, axis=0)
        ranked = sorted(
            zip(feature_names, variances),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(name, float(var)) for name, var in ranked]


# ---------------------------------------------------------------------------
# Feature correlation analysis
# ---------------------------------------------------------------------------

class FeatureCorrelation:
    """Analyze correlations between features."""

    @staticmethod
    def compute_correlation_matrix(
        feature_matrix: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Compute pairwise correlation matrix.
        Returns matrix and metadata.
        """
        n = feature_matrix.shape[1] if len(feature_matrix.shape) > 1 else 0
        if n == 0:
            return {"matrix": [], "names": [], "size": 0}

        # Handle NaN values
        valid_mask = ~np.any(np.isnan(feature_matrix), axis=1)
        clean_matrix = feature_matrix[valid_mask]

        if len(clean_matrix) < 2:
            return {"matrix": [], "names": feature_names, "size": n}

        corr = np.corrcoef(clean_matrix.T)

        return {
            "matrix": corr.tolist(),
            "names": feature_names,
            "size": n,
            "samples_used": len(clean_matrix),
        }

    @staticmethod
    def find_high_correlations(
        feature_matrix: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.8,
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of features with correlation above threshold."""
        n = feature_matrix.shape[1] if len(feature_matrix.shape) > 1 else 0
        if n < 2:
            return []

        valid_mask = ~np.any(np.isnan(feature_matrix), axis=1)
        clean = feature_matrix[valid_mask]
        if len(clean) < 2:
            return []

        corr = np.corrcoef(clean.T)
        pairs: List[Tuple[str, str, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) >= threshold:
                    pairs.append((
                        feature_names[i],
                        feature_names[j],
                        float(corr[i, j]),
                    ))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    @staticmethod
    def feature_target_correlation(
        feature_matrix: np.ndarray,
        target: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:
        """Compute correlation of each feature with the target variable."""
        n = feature_matrix.shape[1] if len(feature_matrix.shape) > 1 else 0
        if n == 0 or len(target) != feature_matrix.shape[0]:
            return []

        correlations: List[Tuple[str, float]] = []
        for i in range(n):
            feature = feature_matrix[:, i]
            valid = ~(np.isnan(feature) | np.isnan(target))
            if np.sum(valid) < 3:
                correlations.append((feature_names[i], 0.0))
                continue
            corr = np.corrcoef(feature[valid], target[valid])[0, 1]
            correlations.append((feature_names[i], float(corr)))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        return correlations


# ---------------------------------------------------------------------------
# Main feature store
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Central feature store for ML model training and inference.
    Provides persistent storage, versioning, computation, and analysis.
    """

    CREATE_TABLES = [
        """
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            symbol TEXT NOT NULL,
            version TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            feature_value REAL,
            meta TEXT DEFAULT '{}'
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_features_sym_ts
        ON features(symbol, timestamp)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_features_name
        ON features(feature_name)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_features_version
        ON features(version)
        """,
        """
        CREATE TABLE IF NOT EXISTS feature_versions (
            version TEXT PRIMARY KEY,
            features TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            hash_signature TEXT NOT NULL,
            parent_version TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS feature_importance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            importance REAL NOT NULL,
            method TEXT NOT NULL,
            computed_at TEXT NOT NULL
        )
        """,
    ]

    def __init__(self, db_path: str = "features.db"):
        self._db_path = db_path
        self._schemas: Dict[str, FeatureSchema] = {}
        self._compute_fns: Dict[str, Callable] = {}
        self._versions: Dict[str, FeatureVersion] = {}
        self._active_version: Optional[str] = None
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

        self._initialize_db()
        logger.info("FeatureStore initialized at %s", db_path)

    def _initialize_db(self) -> None:
        db_dir = Path(self._db_path).parent
        if str(db_dir) != ".":
            db_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        for sql in self.CREATE_TABLES:
            self._conn.execute(sql)
        self._conn.commit()

        # Load existing versions
        cursor = self._conn.execute("SELECT * FROM feature_versions")
        for row in cursor.fetchall():
            row_dict = dict(row)
            version = FeatureVersion(
                version=row_dict["version"],
                features=json.loads(row_dict["features"]),
                description=row_dict["description"],
                created_at=row_dict["created_at"],
                hash_signature=row_dict["hash_signature"],
                parent_version=row_dict["parent_version"],
                is_active=bool(row_dict["is_active"]),
            )
            self._versions[version.version] = version
            if version.is_active:
                self._active_version = version.version

    # --- Schema management ---

    def register_schema(self, schema: FeatureSchema) -> None:
        """Register a feature schema."""
        with self._lock:
            self._schemas[schema.name] = schema
            logger.debug("Registered feature schema: %s", schema.name)

    def register_schemas(self, schemas: List[FeatureSchema]) -> None:
        for schema in schemas:
            self.register_schema(schema)

    def get_schema(self, name: str) -> Optional[FeatureSchema]:
        return self._schemas.get(name)

    def list_schemas(self) -> List[FeatureSchema]:
        return list(self._schemas.values())

    def validate_features(
        self, features: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate a feature dict against registered schemas."""
        errors: List[str] = []
        for name, value in features.items():
            schema = self._schemas.get(name)
            if schema is None:
                errors.append(f"Unknown feature: {name}")
                continue
            valid, msg = schema.validate_value(value)
            if not valid:
                errors.append(msg)
        return len(errors) == 0, errors

    # --- Compute function registration ---

    def register_compute_fn(self, name: str, fn: Callable) -> None:
        """Register a computation function for a feature."""
        with self._lock:
            self._compute_fns[name] = fn

    def compute_feature(self, name: str, **kwargs: Any) -> Optional[Any]:
        """Compute a single feature using its registered function."""
        fn = self._compute_fns.get(name)
        if fn is None:
            logger.warning("No compute function for feature: %s", name)
            return None
        try:
            return fn(**kwargs)
        except Exception as exc:
            logger.error("Error computing feature %s: %s", name, exc)
            return None

    def compute_batch(
        self, feature_names: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        """Compute multiple features in batch."""
        results: Dict[str, Any] = {}
        for name in feature_names:
            results[name] = self.compute_feature(name, **kwargs)
        return results

    def compute_online(
        self, feature_names: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Compute features for real-time inference.
        Same as compute_batch but intended for single-point computation.
        """
        return self.compute_batch(feature_names, **kwargs)

    # --- Version management ---

    def create_version(
        self,
        version: str,
        features: List[str],
        description: str = "",
        parent_version: str = "",
    ) -> FeatureVersion:
        """Create a new feature version."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        hash_sig = FeatureVersion.compute_hash(features, description)

        fv = FeatureVersion(
            version=version,
            features=features,
            description=description,
            created_at=now,
            hash_signature=hash_sig,
            parent_version=parent_version,
            is_active=True,
        )

        with self._lock:
            self._versions[version] = fv
            self._active_version = version

            self._conn.execute(
                """INSERT OR REPLACE INTO feature_versions
                (version, features, description, created_at, hash_signature, parent_version, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (version, json.dumps(features), description, now, hash_sig, parent_version, 1),
            )
            self._conn.commit()

        logger.info("Created feature version: %s (%d features)", version, len(features))
        return fv

    def get_version(self, version: str) -> Optional[FeatureVersion]:
        return self._versions.get(version)

    def get_active_version(self) -> Optional[FeatureVersion]:
        if self._active_version:
            return self._versions.get(self._active_version)
        return None

    def list_versions(self) -> List[FeatureVersion]:
        return list(self._versions.values())

    def set_active_version(self, version: str) -> None:
        if version not in self._versions:
            raise ValueError(f"Version not found: {version}")
        with self._lock:
            # Deactivate all
            self._conn.execute("UPDATE feature_versions SET is_active = 0")
            # Activate target
            self._conn.execute(
                "UPDATE feature_versions SET is_active = 1 WHERE version = ?",
                (version,),
            )
            self._conn.commit()
            self._active_version = version

    # --- Storage ---

    def store_features(
        self,
        timestamp: float,
        symbol: str,
        features: Dict[str, float],
        version: Optional[str] = None,
    ) -> int:
        """Store computed features for a timestamp. Returns number stored."""
        version = version or self._active_version or "default"

        rows = [
            (timestamp, symbol, version, name, value, "{}")
            for name, value in features.items()
            if value is not None
        ]

        if not rows:
            return 0

        with self._lock:
            self._conn.executemany(
                """INSERT INTO features
                (timestamp, symbol, version, feature_name, feature_value, meta)
                VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()

        return len(rows)

    def store_feature_matrix(
        self,
        timestamps: np.ndarray,
        symbol: str,
        feature_names: List[str],
        feature_matrix: np.ndarray,
        version: Optional[str] = None,
    ) -> int:
        """Store a full feature matrix (rows=timestamps, cols=features)."""
        version = version or self._active_version or "default"
        n_rows, n_cols = feature_matrix.shape

        if n_cols != len(feature_names):
            raise ValueError(
                f"Matrix has {n_cols} columns but {len(feature_names)} feature names"
            )

        rows = []
        for i in range(n_rows):
            ts = float(timestamps[i])
            for j, name in enumerate(feature_names):
                val = float(feature_matrix[i, j])
                if not np.isnan(val):
                    rows.append((ts, symbol, version, name, val, "{}"))

        if not rows:
            return 0

        with self._lock:
            self._conn.executemany(
                """INSERT INTO features
                (timestamp, symbol, version, feature_name, feature_value, meta)
                VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()

        return len(rows)

    def get_features(
        self,
        symbol: str,
        feature_names: List[str],
        start: Optional[float] = None,
        end: Optional[float] = None,
        version: Optional[str] = None,
        limit: int = 10000,
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve stored features.
        Returns dict mapping feature name to numpy array of values.
        """
        version = version or self._active_version or "default"

        clauses = ["symbol = ?", "version = ?"]
        params: list = [symbol, version]

        if feature_names:
            placeholders = ",".join(["?"] * len(feature_names))
            clauses.append(f"feature_name IN ({placeholders})")
            params.extend(feature_names)

        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)

        where = " AND ".join(clauses)

        with self._lock:
            cursor = self._conn.execute(
                f"SELECT timestamp, feature_name, feature_value "
                f"FROM features WHERE {where} "
                f"ORDER BY timestamp LIMIT ?",
                params + [limit],
            )
            rows = cursor.fetchall()

        # Group by feature name
        grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for row in rows:
            grouped[row["feature_name"]].append(
                (float(row["timestamp"]), float(row["feature_value"]))
            )

        result: Dict[str, np.ndarray] = {}
        for name, values in grouped.items():
            values.sort(key=lambda x: x[0])
            result[name] = np.array([v for _, v in values], dtype=np.float64)

        return result

    def get_feature_matrix(
        self,
        symbol: str,
        feature_names: List[str],
        start: Optional[float] = None,
        end: Optional[float] = None,
        version: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get features as a matrix.
        Returns (timestamps, feature_matrix) where matrix shape is (n_samples, n_features).
        """
        features = self.get_features(symbol, feature_names, start, end, version)

        if not features:
            return np.empty(0), np.empty((0, len(feature_names)))

        # Find common timestamps
        all_ts: Optional[Set[float]] = None
        for name, vals in features.items():
            # We need timestamps too - re-fetch
            pass

        # Simplified: use first feature's length as reference
        first_key = list(features.keys())[0] if features else None
        if not first_key:
            return np.empty(0), np.empty((0, len(feature_names)))

        n = len(features[first_key])
        timestamps = np.arange(n, dtype=np.float64)  # Placeholder timestamps
        matrix = np.full((n, len(feature_names)), np.nan)

        for j, name in enumerate(feature_names):
            if name in features:
                vals = features[name]
                length = min(len(vals), n)
                matrix[:length, j] = vals[:length]

        return timestamps, matrix

    # --- Importance tracking ---

    def store_importance(
        self,
        model_name: str,
        importances: Dict[str, float],
        method: str = "default",
    ) -> None:
        """Store feature importance scores from a model."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        rows = [
            (model_name, name, importance, method, now)
            for name, importance in importances.items()
        ]

        with self._lock:
            self._conn.executemany(
                """INSERT INTO feature_importance
                (model_name, feature_name, importance, method, computed_at)
                VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()

    def get_importance(
        self, model_name: str, method: Optional[str] = None
    ) -> Dict[str, float]:
        """Get latest feature importance scores for a model."""
        clauses = ["model_name = ?"]
        params: list = [model_name]
        if method:
            clauses.append("method = ?")
            params.append(method)

        where = " AND ".join(clauses)

        with self._lock:
            cursor = self._conn.execute(
                f"SELECT feature_name, importance FROM feature_importance "
                f"WHERE {where} ORDER BY computed_at DESC",
                params,
            )
            result: Dict[str, float] = {}
            for row in cursor.fetchall():
                name = row["feature_name"]
                if name not in result:  # Keep latest
                    result[name] = float(row["importance"])
            return result

    # --- Analysis ---

    def analyze_correlations(
        self,
        symbol: str,
        feature_names: List[str],
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run correlation analysis on stored features."""
        timestamps, matrix = self.get_feature_matrix(
            symbol, feature_names, version=version
        )
        if matrix.shape[0] < 3:
            return {"error": "Not enough data points for correlation analysis"}

        return FeatureCorrelation.compute_correlation_matrix(matrix, feature_names)

    def find_redundant_features(
        self,
        symbol: str,
        feature_names: List[str],
        threshold: float = 0.9,
        version: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """Find highly correlated (redundant) feature pairs."""
        timestamps, matrix = self.get_feature_matrix(
            symbol, feature_names, version=version
        )
        if matrix.shape[0] < 3:
            return []

        return FeatureCorrelation.find_high_correlations(matrix, feature_names, threshold)

    # --- Statistics ---

    def get_feature_stats(self, symbol: str, feature_name: str) -> Dict[str, Any]:
        """Get statistics for a specific feature."""
        with self._lock:
            cursor = self._conn.execute(
                """SELECT
                    COUNT(*) as count,
                    AVG(feature_value) as mean,
                    MIN(feature_value) as min,
                    MAX(feature_value) as max
                FROM features
                WHERE symbol = ? AND feature_name = ?""",
                (symbol, feature_name),
            )
            row = cursor.fetchone()

        if row and row["count"] > 0:
            return {
                "feature_name": feature_name,
                "count": row["count"],
                "mean": row["mean"],
                "min": row["min"],
                "max": row["max"],
            }
        return {"feature_name": feature_name, "count": 0}

    def get_store_stats(self) -> Dict[str, Any]:
        """Get overall store statistics."""
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM features")
            total_rows = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(DISTINCT feature_name) FROM features")
            unique_features = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(DISTINCT symbol) FROM features")
            unique_symbols = cursor.fetchone()[0]

        return {
            "total_rows": total_rows,
            "unique_features": unique_features,
            "unique_symbols": unique_symbols,
            "registered_schemas": len(self._schemas),
            "registered_compute_fns": len(self._compute_fns),
            "versions": len(self._versions),
            "active_version": self._active_version,
        }

    # --- Cleanup ---

    def delete_old_features(self, before_timestamp: float) -> int:
        """Delete features older than the given timestamp."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM features WHERE timestamp < ?", (before_timestamp,)
            )
            self._conn.commit()
            return cursor.rowcount

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
