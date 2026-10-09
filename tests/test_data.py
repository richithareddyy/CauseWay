"""Checksum-verified acquisition and raw-data immutability."""

import dataclasses
import json
import os
import stat

import pytest

from causeway.config import Source, load_data_config
from causeway.data import (
    AcquisitionError,
    ChecksumError,
    acquire_raw,
    raw_path,
    read_raw,
    sha256_file,
)


def test_download_is_verified_and_made_read_only(data_config, paths, source_file):
    path, digest = acquire_raw(data_config, paths)[0], source_file[1]
    assert sha256_file(path) == digest
    assert not os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    provenance = json.loads(path.with_name(path.name + ".source.json").read_text())
    assert provenance["source"] == "local" and provenance["sha256"] == digest


def test_checksum_mismatch_is_rejected_and_leaves_no_file(data_config, paths):
    bad = dataclasses.replace(
        data_config, sources=[dataclasses.replace(data_config.sources[0], sha256="0" * 64)]
    )
    with pytest.raises(AcquisitionError, match="expected sha256"):
        acquire_raw(bad, paths)
    assert not raw_path(bad, paths).exists()


def test_falls_back_to_mirror_and_records_the_failed_source(data_config, paths, gzip_source, tmp_path):
    missing = Source(name="unreachable", url=(tmp_path / "absent.csv").as_uri(), sha256="0" * 64)
    config = dataclasses.replace(data_config, sources=[missing, gzip_source])
    path, provenance = acquire_raw(config, paths)
    assert provenance["source"] == "local_mirror"
    assert provenance["failed_sources"] and provenance["failed_sources"][0].startswith("unreachable")
    assert sha256_file(path) == gzip_source.sha256


def test_corrupted_compressed_mirror_is_rejected(data_config, paths, gzip_source):
    corrupted = dataclasses.replace(gzip_source, sha256_compressed="f" * 64)
    with pytest.raises(AcquisitionError, match="compressed checksum mismatch"):
        acquire_raw(dataclasses.replace(data_config, sources=[corrupted]), paths)


def test_existing_raw_file_is_reused_not_downloaded_again(data_config, paths):
    first, _ = acquire_raw(data_config, paths)
    mtime = first.stat().st_mtime_ns
    unreachable = dataclasses.replace(data_config.sources[0], url="file:///nonexistent/never-read.csv")
    second, provenance = acquire_raw(dataclasses.replace(data_config, sources=[unreachable]), paths)
    assert second == first and second.stat().st_mtime_ns == mtime
    assert provenance["source"] == "local"


def test_modified_raw_file_raises_instead_of_being_overwritten(data_config, paths):
    path, _ = acquire_raw(data_config, paths)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.write_bytes(path.read_bytes().replace(b"Phone", b"Fax", 1))
    with pytest.raises(ChecksumError, match="modified after download"):
        acquire_raw(data_config, paths)
    assert b"Fax" in path.read_bytes()  # the evidence is left in place for inspection


def test_read_raw_keeps_every_field_as_text_without_nan_conversion(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("recency,spend\n1,0\n2,\n3,NA\n")
    frame = read_raw(path)
    assert frame["spend"].tolist() == ["0", "", "NA"]
    assert frame.isna().sum().sum() == 0


def test_project_data_config_lists_original_source_first():
    config = load_data_config()
    assert config.expected_rows == 64000
    assert config.sources[0].name == "minethatdata_original"
    assert config.sources[0].sha256 == "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
    assert config.sources[1].compression == "gzip"
