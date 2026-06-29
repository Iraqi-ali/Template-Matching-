"""
Metadata & EXIF Analysis Module
================================
Extracts and analyzes document metadata for forgery detection.

Capabilities:
  - EXIF data extraction (camera, timestamp, GPS)
  - Creation/modification date analysis
  - Software/hardware fingerprinting
  - File structure analysis (JFIF, PNG chunks, etc.)
  - Anomaly detection in metadata
  - Timestamp consistency checking
  - GPS location verification
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import os
import json
import struct
from datetime import datetime
from enum import Enum


class MetadataAnomaly(Enum):
    NONE = "No anomalies"
    TIMESTAMP_MISMATCH = "Creation/modification timestamp inconsistency"
    SOFTWARE_SUSPICIOUS = "Suspicious editing software detected"
    GPS_INCONSISTENT = "GPS data inconsistent with expected location"
    CAMERA_MISMATCH = "Camera make/model differs from expected"
    RESOLUTION_ANOMALY = "Unusual resolution for claimed device"
    COMPRESSION_ANOMALY = "Compression artifacts inconsistent with claimed origin"
    FILE_STRUCTURE = "File structure anomalies detected"
    METADATA_STRIPPED = "Metadata intentionally stripped (potential anti-forensics)"


@dataclass
class MetadataReport:
    """Complete metadata analysis report."""
    file_path: str = ""
    file_name: str = ""
    file_size_bytes: int = 0
    file_extension: str = ""
    mime_type: str = ""

    # EXIF
    has_exif: bool = False
    exif_data: Dict[str, Any] = field(default_factory=dict)

    # Dates
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    date_original: Optional[str] = None

    # Camera
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software_used: Optional[str] = None

    # GPS
    has_gps: bool = False
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None

    # Image properties
    image_width: int = 0
    image_height: int = 0
    color_space: str = ""
    compression: str = ""
    resolution_dpi: Tuple[float, float] = (0, 0)

    # Anomalies
    anomalies: List[MetadataAnomaly] = field(default_factory=list)
    anomaly_score: float = 0.0
    is_suspicious: bool = False
    details: List[str] = field(default_factory=list)


class MetadataAnalyzer:
    """
    Document metadata forensic analyzer.

    Usage:
        analyzer = MetadataAnalyzer()
        report = analyzer.analyze("document.jpg")
        print(f"Anomalies: {len(report.anomalies)}")
    """

    def analyze(self, file_path: str, expected_meta: Optional[Dict] = None) -> MetadataReport:
        """
        Analyze file metadata for forgery indicators.

        Args:
            file_path: Path to the image file.
            expected_meta: Optional dict of expected values (e.g., {'camera': 'Canon EOS'})
                           to compare against.

        Returns:
            MetadataReport with findings.
        """
        report = MetadataReport()
        report.file_path = file_path
        report.file_name = Path(file_path).name
        report.file_extension = Path(file_path).suffix.lower()

        if not os.path.exists(file_path):
            report.details.append("File not found")
            return report

        # File stats
        stat = os.stat(file_path)
        report.file_size_bytes = stat.st_size
        report.date_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
        report.date_created = datetime.fromtimestamp(stat.st_ctime).isoformat()

        # Read magic bytes for MIME type
        report.mime_type = self._detect_mime_type(file_path)

        # Extract EXIF if available
        self._extract_exif(file_path, report)

        # Read image dimensions directly with OpenCV
        img = cv2.imread(file_path)
        if img is not None:
            report.image_height, report.image_width = img.shape[:2]

        # Analyze for anomalies
        self._detect_anomalies(report, expected_meta)
        report.anomaly_score = min(len(report.anomalies) * 0.2, 1.0)
        report.is_suspicious = len(report.anomalies) > 0
        report.details = self._generate_details(report)

        return report

    def compare_metadata(
        self, file1: str, file2: str
    ) -> Dict[str, Any]:
        """Compare metadata between two files for inconsistencies."""
        r1 = self.analyze(file1)
        r2 = self.analyze(file2)

        differences = []

        if r1.camera_make != r2.camera_make:
            differences.append(f"Camera make: '{r1.camera_make}' vs '{r2.camera_make}'")
        if r1.camera_model != r2.camera_model:
            differences.append(f"Camera model: '{r1.camera_model}' vs '{r2.camera_model}'")
        if r1.software_used != r2.software_used:
            differences.append(f"Software: '{r1.software_used}' vs '{r2.software_used}'")
        if abs(r1.image_width - r2.image_width) > 10 or abs(r1.image_height - r2.image_height) > 10:
            differences.append(f"Resolution: {r1.image_width}x{r1.image_height} vs {r2.image_width}x{r2.image_height}")
        if r1.mime_type != r2.mime_type:
            differences.append(f"Format: '{r1.mime_type}' vs '{r2.mime_type}'")

        return {
            "differences": differences,
            "consistent": len(differences) == 0,
            "file1_anomalies": [a.name for a in r1.anomalies],
            "file2_anomalies": [a.name for a in r2.anomalies],
            "file1_score": r1.anomaly_score,
            "file2_score": r2.anomaly_score,
        }

    # ==================================================================
    # Internal Methods
    # ==================================================================

    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type from magic bytes."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(12)
            if header.startswith(b'\xff\xd8\xff'):
                return 'image/jpeg'
            elif header.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'image/png'
            elif header.startswith(b'GIF8'):
                return 'image/gif'
            elif header.startswith(b'BM'):
                return 'image/bmp'
            elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
                return 'image/tiff'
            return 'application/octet-stream'
        except Exception:
            return 'unknown'

    def _extract_exif(self, file_path: str, report: MetadataReport):
        """Extract EXIF metadata from image file."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS

            img = Image.open(file_path)
            exif_data = img._getexif()

            if exif_data is None:
                report.has_exif = False
                report.anomalies.append(MetadataAnomaly.METADATA_STRIPPED)
                return

            report.has_exif = True

            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                report.exif_data[tag_name] = str(value) if not isinstance(value, (int, float, str, bytes)) else value

                if tag_name == 'DateTimeOriginal':
                    report.date_original = str(value)
                elif tag_name == 'Make':
                    report.camera_make = str(value).strip('\x00')
                elif tag_name == 'Model':
                    report.camera_model = str(value).strip('\x00')
                elif tag_name == 'Software':
                    report.software_used = str(value).strip('\x00')
                elif tag_name == 'ColorSpace':
                    report.color_space = str(value)
                elif tag_name == 'Compression':
                    report.compression = str(value)
                elif tag_name in ('XResolution', 'YResolution'):
                    try:
                        if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                            dpi = float(value.numerator) / float(value.denominator)
                            if tag_name == 'XResolution':
                                report.resolution_dpi = (dpi, report.resolution_dpi[1])
                            else:
                                report.resolution_dpi = (report.resolution_dpi[0], dpi)
                    except Exception:
                        pass

            # GPS extraction
            gps_info = {}
            if hasattr(img, '_getexif'):
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        if TAGS.get(tag_id) == 'GPSInfo':
                            for gps_tag_id, gps_value in value.items():
                                gps_tag = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                                gps_info[gps_tag] = gps_value
                            report.has_gps = True

                            # Parse GPS coordinates
                            if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                                report.gps_latitude = self._parse_gps_coord(
                                    gps_info['GPSLatitude'],
                                    gps_info.get('GPSLatitudeRef', 'N')
                                )
                            if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                                report.gps_longitude = self._parse_gps_coord(
                                    gps_info['GPSLongitude'],
                                    gps_info.get('GPSLongitudeRef', 'E')
                                )

            report.exif_data['gps_info'] = {k: str(v) for k, v in gps_info.items()}

        except ImportError:
            report.details.append("PIL not available for EXIF extraction")
        except Exception as e:
            report.details.append(f"EXIF extraction error: {str(e)}")

    def _parse_gps_coord(self, coord, ref: str) -> float:
        """Parse GPS coordinate from EXIF format."""
        try:
            degrees = float(coord[0])
            minutes = float(coord[1])
            seconds = float(coord[2])
            decimal = degrees + minutes / 60.0 + seconds / 3600.0
            if ref in ('S', 'W'):
                decimal = -decimal
            return decimal
        except Exception:
            return 0.0

    def _detect_anomalies(self, report: MetadataReport, expected: Optional[Dict]):
        """Detect metadata anomalies."""
        # Timestamp consistency
        if report.date_original and report.date_modified:
            try:
                dt_orig = datetime.strptime(report.date_original, '%Y:%m:%d %H:%M:%S')
                dt_mod = datetime.fromisoformat(report.date_modified)
                if abs((dt_mod - dt_orig).total_seconds()) > 86400 * 7:  # > 1 week
                    report.anomalies.append(MetadataAnomaly.TIMESTAMP_MISMATCH)
            except ValueError:
                pass

        # Suspicious software
        editing_software = ['photoshop', 'gimp', 'lightroom', 'affinity', 'paint.net', 'canva']
        if report.software_used:
            sw_lower = report.software_used.lower()
            if any(es in sw_lower for es in editing_software):
                report.anomalies.append(MetadataAnomaly.SOFTWARE_SUSPICIOUS)

        # Resolution anomaly
        if report.camera_make and report.camera_model:
            # Most smartphone cameras produce 4000+ px wide images
            if report.image_width < 1000 and report.image_width > 0:
                report.anomalies.append(MetadataAnomaly.RESOLUTION_ANOMALY)

        # Compare against expected
        if expected:
            if 'camera' in expected and report.camera_make:
                if expected['camera'].lower() not in (report.camera_make or '').lower():
                    report.anomalies.append(MetadataAnomaly.CAMERA_MISMATCH)
            if 'date' in expected and report.date_original:
                if expected['date'] != report.date_original:
                    report.anomalies.append(MetadataAnomaly.TIMESTAMP_MISMATCH)
            if 'gps_lat' in expected and report.has_gps:
                lat_diff = abs((expected['gps_lat'] or 0) - (report.gps_latitude or 0))
                if lat_diff > 0.01:
                    report.anomalies.append(MetadataAnomaly.GPS_INCONSISTENT)

    def _generate_details(self, report: MetadataReport) -> List[str]:
        """Generate human-readable details."""
        details = []

        details.append(f"📄 File: {report.file_name}")
        details.append(f"   Size: {report.file_size_bytes:,} bytes | Type: {report.mime_type}")
        details.append(f"   Resolution: {report.image_width}×{report.image_height}")

        if report.has_exif:
            details.append(f"📷 Camera: {report.camera_make or 'N/A'} {report.camera_model or ''}")
            details.append(f"   Software: {report.software_used or 'N/A'}")
            details.append(f"   Date Original: {report.date_original or 'N/A'}")
            if report.has_gps:
                details.append(f"📍 GPS: {report.gps_latitude:.4f}, {report.gps_longitude:.4f}")
        else:
            details.append("⚠ No EXIF metadata found (possibly stripped)")

        if report.anomalies:
            details.append(f"\n🔴 ANOMALIES DETECTED ({len(report.anomalies)}):")
            for anomaly in report.anomalies:
                details.append(f"   • {anomaly.value}")
        else:
            details.append("\n✅ No metadata anomalies detected")

        return details
