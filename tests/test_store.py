from synthetic import build_store, synthetic_queries

from steerdb.store import ExperienceStore


def test_backup_is_a_complete_readable_copy(tmp_path):
    store = build_store(synthetic_queries(2, "a"))
    dst = tmp_path / "drive" / "experience.sqlite"
    store.backup_to(dst)
    store.backup_to(dst)  # overwriting an existing backup works too
    copy = ExperienceStore(dst)
    assert len(copy.observations()) == len(store.observations()) == 16
    assert copy.bootstrap_table().keys() == {"1a", "2a"}
    assert not (tmp_path / "drive" / "experience.sqlite.tmp").exists()
