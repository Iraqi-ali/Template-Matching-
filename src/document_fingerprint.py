"""
Document Fingerprint Vault — Secure Image Identity Storage
===========================================================
Stores perceptual hashes and cryptographic fingerprints of source documents.
Enables instant verification without re-uploading the original.

Features:
  - Perceptual Hash (pHash) — visually similar images match
  - Average Hash (aHash) — fast comparison
  - Difference Hash (dHash) — gradient-based
  - Color Hash — color distribution fingerprint
  - SHA-256 — cryptographic integrity
  - SQLite vault — persistent storage with metadata
  - Similarity Search — find closest match in vault
  - Tamper-proof audit log
"""

import cv2
import numpy as np
import hashlib
import sqlite3
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from datetime import datetime


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class DocumentFingerprint:
    """Complete fingerprint of a document image."""
    # Identity
    fingerprint_id: str = ""          # UUID
    label: str = ""                   # Human-readable name
    file_path: str = ""               # Original file path
    original_filename: str = ""

    # Hashes
    phash: str = ""                   # Perceptual hash (64-bit hex)
    ahash: str = ""                   # Average hash (64-bit hex)
    dhash: str = ""                   # Difference hash (64-bit hex)
    chash: str = ""                   # Color hash (192-bit hex)
    sha256: str = ""                  # SHA-256 of raw bytes

    # Metadata
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    image_format: str = ""
    created_at: str = ""              # ISO timestamp
    stored_at: str = ""               # ISO timestamp

    # Statistics
    mean_brightness: float = 0.0
    std_brightness: float = 0.0
    edge_density: float = 0.0
    histogram_bins: str = ""          # JSON-serialized histogram

    # Extras
    tags: str = ""                    # Comma-separated
    notes: str = ""


@dataclass
class MatchResult:
    """Result of a fingerprint match search."""
    fingerprint: DocumentFingerprint
    phash_distance: int = 0           # Hamming distance (0 = identical)
    ahash_distance: int = 0
    dhash_distance: int = 0
    similarity_score: float = 1.0     # 1.0 = identical, 0.0 = completely different
    is_match: bool = False


# ===========================================================================
# Hash Functions
# ===========================================================================

def compute_phash(img: np.ndarray) -> str:
    """Perceptual hash (pHash) using DCT. Robust to minor changes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (32, 32))
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]
    mean_val = np.mean(dct_low)
    bits = (dct_low > mean_val).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def compute_ahash(img: np.ndarray) -> str:
    """Average hash. Fast but sensitive to brightness changes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (8, 8))
    mean_val = np.mean(resized)
    bits = (resized > mean_val).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def compute_dhash(img: np.ndarray) -> str:
    """Difference hash. Captures gradient differences."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (9, 8))
    bits = (resized[:, 1:] > resized[:, :-1]).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def compute_chash(img: np.ndarray) -> str:
    """Color hash. Captures color distribution."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(img, (8, 8))
    # Per-channel mean comparison
    parts = []
    for ch in range(3):
        channel = resized[:, :, ch]
        mean_val = np.mean(channel)
        bits = (channel > mean_val).flatten()
        hash_int = 0
        for bit in bits:
            hash_int = (hash_int << 1) | int(bit)
        parts.append(f"{hash_int:016x}")
    return "".join(parts)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    if len(hash1) != len(hash2):
        max_len = max(len(hash1), len(hash2))
        hash1 = hash1.zfill(max_len)
        hash2 = hash2.zfill(max_len)
    try:
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        xor = int1 ^ int2
        return bin(xor).count("1")
    except ValueError:
        return 999


# ===========================================================================
# Fingerprint Vault (SQLite)
# ===========================================================================

class FingerprintVault:
    """
    Secure document fingerprint storage.

    Usage:
        vault = FingerprintVault("vault.db")
        fp = vault.register("passport_original.png", "My Passport")
        results = vault.search(suspect_image)
    """

    def __init__(self, db_path: str = "fingerprint_vault.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    fingerprint_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    file_path TEXT,
                    original_filename TEXT,
                    phash TEXT NOT NULL,
                    ahash TEXT NOT NULL,
                    dhash TEXT NOT NULL,
                    chash TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    file_size_bytes INTEGER,
                    image_format TEXT,
                    mean_brightness REAL,
                    std_brightness REAL,
                    edge_density REAL,
                    histogram_bins TEXT,
                    tags TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    stored_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    fingerprint_id TEXT,
                    details TEXT,
                    timestamp TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()

    # ==================================================================
    # Registration
    # ==================================================================

    def register(
        self,
        image: np.ndarray,
        label: str,
        file_path: str = "",
        tags: str = "",
        notes: str = "",
    ) -> DocumentFingerprint:
        """
        Register a source document fingerprint in the vault.

        Args:
            image: The document image (BGR numpy array).
            label: Human-readable label (e.g., "Passport Page 1").
            file_path: Optional original file path.
            tags: Comma-separated tags.
            notes: Free-text notes.

        Returns:
            DocumentFingerprint with all computed hashes.
        """
        import uuid

        fp = DocumentFingerprint()
        fp.fingerprint_id = str(uuid.uuid4())
        fp.label = label
        fp.file_path = file_path
        fp.original_filename = Path(file_path).name if file_path else ""
        fp.tags = tags
        fp.notes = notes

        # Compute hashes
        fp.phash = compute_phash(image)
        fp.ahash = compute_ahash(image)
        fp.dhash = compute_dhash(image)
        fp.chash = compute_chash(image)

        # SHA-256 (from raw bytes)
        _, buf = cv2.imencode('.png', image)
        fp.sha256 = hashlib.sha256(buf.tobytes()).hexdigest()

        # Metadata
        fp.height, fp.width = image.shape[:2]
        fp.image_format = "PNG"
        fp.stored_at = datetime.now().isoformat()

        # Statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        fp.mean_brightness = float(np.mean(gray))
        fp.std_brightness = float(np.std(gray))

        edges = cv2.Canny(gray, 50, 150)
        fp.edge_density = float(np.sum(edges > 0) / edges.size)

        # Histogram (64 bins)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        fp.histogram_bins = json.dumps([float(h[0]) for h in hist])

        # Store in DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO fingerprints (
                    fingerprint_id, label, file_path, original_filename,
                    phash, ahash, dhash, chash, sha256,
                    width, height, image_format,
                    mean_brightness, std_brightness, edge_density, histogram_bins,
                    tags, notes, created_at, stored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fp.fingerprint_id, fp.label, fp.file_path, fp.original_filename,
                fp.phash, fp.ahash, fp.dhash, fp.chash, fp.sha256,
                fp.width, fp.height, fp.image_format,
                fp.mean_brightness, fp.std_brightness, fp.edge_density, fp.histogram_bins,
                fp.tags, fp.notes, fp.stored_at, fp.stored_at,
            ))
            conn.execute("""
                INSERT INTO audit_log (action, fingerprint_id, details)
                VALUES (?, ?, ?)
            """, ("REGISTER", fp.fingerprint_id, f"Registered: {label}"))
            conn.commit()

        return fp

    def register_from_file(self, file_path: str, label: str = "", tags: str = "") -> DocumentFingerprint:
        """Register a document from a file path."""
        from .utils import load_image
        img = load_image(file_path)
        if not label:
            label = Path(file_path).stem
        return self.register(img, label=label, file_path=file_path, tags=tags)

    # ==================================================================
    # Search & Verify
    # ==================================================================

    def search(
        self,
        suspect_image: np.ndarray,
        max_results: int = 5,
        phash_threshold: int = 10,
        ahash_threshold: int = 10,
    ) -> List[MatchResult]:
        """
        Search the vault for the closest matching fingerprint.

        Args:
            suspect_image: The image to search for.
            max_results: Max results to return.
            phash_threshold: Max Hamming distance for pHash match.
            ahash_threshold: Max Hamming distance for aHash match.

        Returns:
            List of MatchResult, best match first.
        """
        # Compute suspect hashes
        s_phash = compute_phash(suspect_image)
        s_ahash = compute_ahash(suspect_image)
        s_dhash = compute_dhash(suspect_image)

        results = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM fingerprints ORDER BY stored_at DESC"
            ).fetchall()

        for row in rows:
            fp = self._row_to_fingerprint(row)
            phash_dist = hamming_distance(s_phash, fp.phash)
            ahash_dist = hamming_distance(s_ahash, fp.ahash)
            dhash_dist = hamming_distance(s_dhash, fp.dhash)

            # Similarity score (weighted average)
            phash_sim = max(0.0, 1.0 - phash_dist / 64.0)
            ahash_sim = max(0.0, 1.0 - ahash_dist / 64.0)
            dhash_sim = max(0.0, 1.0 - dhash_dist / 64.0)
            similarity = phash_sim * 0.5 + ahash_sim * 0.2 + dhash_sim * 0.3

            is_match = phash_dist <= phash_threshold and ahash_dist <= ahash_threshold

            results.append(MatchResult(
                fingerprint=fp,
                phash_distance=phash_dist,
                ahash_distance=ahash_dist,
                dhash_distance=dhash_dist,
                similarity_score=similarity,
                is_match=is_match,
            ))

        # Sort by similarity (best first)
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:max_results]

    def verify(
        self,
        suspect_image: np.ndarray,
        fingerprint_id: str,
    ) -> MatchResult:
        """Verify a suspect against a specific registered fingerprint."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM fingerprints WHERE fingerprint_id = ?",
                (fingerprint_id,)
            ).fetchone()

        if not row:
            return MatchResult(
                fingerprint=DocumentFingerprint(),
                similarity_score=0.0,
                is_match=False,
            )

        fp = self._row_to_fingerprint(row)
        s_phash = compute_phash(suspect_image)
        s_ahash = compute_ahash(suspect_image)
        s_dhash = compute_dhash(suspect_image)

        phash_dist = hamming_distance(s_phash, fp.phash)
        ahash_dist = hamming_distance(s_ahash, fp.ahash)
        dhash_dist = hamming_distance(s_dhash, fp.dhash)

        phash_sim = max(0.0, 1.0 - phash_dist / 64.0)
        ahash_sim = max(0.0, 1.0 - ahash_dist / 64.0)
        dhash_sim = max(0.0, 1.0 - dhash_dist / 64.0)
        similarity = phash_sim * 0.5 + ahash_sim * 0.2 + dhash_sim * 0.3

        is_match = phash_dist <= 10 and ahash_dist <= 10

        # Log verification
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_log (action, fingerprint_id, details)
                VALUES (?, ?, ?)
            """, ("VERIFY", fingerprint_id,
                  f"Match={is_match}, Similarity={similarity:.4f}, pHashDist={phash_dist}"))
            conn.commit()

        return MatchResult(
            fingerprint=fp,
            phash_distance=phash_dist,
            ahash_distance=ahash_dist,
            dhash_distance=dhash_dist,
            similarity_score=similarity,
            is_match=is_match,
        )

    # ==================================================================
    # Vault Management
    # ==================================================================

    def list_all(self) -> List[DocumentFingerprint]:
        """List all registered fingerprints."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM fingerprints ORDER BY stored_at DESC"
            ).fetchall()
        return [self._row_to_fingerprint(r) for r in rows]

    def get_by_id(self, fingerprint_id: str) -> Optional[DocumentFingerprint]:
        """Get a specific fingerprint by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM fingerprints WHERE fingerprint_id = ?",
                (fingerprint_id,)
            ).fetchone()
        if row:
            return self._row_to_fingerprint(row)
        return None

    def delete(self, fingerprint_id: str) -> bool:
        """Delete a fingerprint from the vault."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM fingerprints WHERE fingerprint_id = ?",
                (fingerprint_id,)
            )
            conn.execute("""
                INSERT INTO audit_log (action, fingerprint_id, details)
                VALUES (?, ?, ?)
            """, ("DELETE", fingerprint_id, "Fingerprint deleted"))
            conn.commit()
        return True

    def count(self) -> int:
        """Return total number of fingerprints in vault."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()
            return row[0] if row else 0

    def get_audit_log(self, limit: int = 50) -> List[Dict]:
        """Get recent audit log entries."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [
            {"id": r[0], "action": r[1], "fingerprint_id": r[2],
             "details": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def compare_two_fingerprints(
        self, fp1_id: str, fp2_id: str
    ) -> Dict:
        """Compare two stored fingerprints in detail."""
        fp1 = self.get_by_id(fp1_id)
        fp2 = self.get_by_id(fp2_id)
        if not fp1 or not fp2:
            return {"error": "One or both fingerprints not found"}

        return {
            "phash_distance": hamming_distance(fp1.phash, fp2.phash),
            "ahash_distance": hamming_distance(fp1.ahash, fp2.ahash),
            "dhash_distance": hamming_distance(fp1.dhash, fp2.dhash),
            "chash_distance": hamming_distance(fp1.chash, fp2.chash),
            "sha256_match": fp1.sha256 == fp2.sha256,
            "brightness_diff": abs(fp1.mean_brightness - fp2.mean_brightness),
            "edge_density_diff": abs(fp1.edge_density - fp2.edge_density),
            "size_match": fp1.width == fp2.width and fp1.height == fp2.height,
        }

    def _row_to_fingerprint(self, row) -> DocumentFingerprint:
        """Convert a DB row to a DocumentFingerprint."""
        cols = [
            "fingerprint_id", "label", "file_path", "original_filename",
            "phash", "ahash", "dhash", "chash", "sha256",
            "width", "height", "file_size_bytes", "image_format",
            "mean_brightness", "std_brightness", "edge_density", "histogram_bins",
            "tags", "notes", "created_at", "stored_at",
        ]
        data = dict(zip(cols, row))
        return DocumentFingerprint(**data)


# ===========================================================================
# Convenience: Global vault instance
# ===========================================================================

_default_vault: Optional[FingerprintVault] = None


def get_vault(db_path: str = "fingerprint_vault.db") -> FingerprintVault:
    """Get or create the default fingerprint vault."""
    global _default_vault
    if _default_vault is None or _default_vault.db_path != db_path:
        _default_vault = FingerprintVault(db_path)
    return _default_vault
