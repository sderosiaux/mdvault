from mdvault.vault import Vault


def test_vault_store_and_search(db_path, mock_embedder):
    """Vault.store() then Vault.search() finds the memory."""
    v = Vault(db_path, embedder=mock_embedder)
    v.store("Python is great for scripting and automation", namespace="lang")

    results = v.search("Python scripting")
    assert len(results) > 0
    assert any("Python" in r.get("raw_content", r.get("content", "")) for r in results)
    v.close()


def test_vault_store_and_delete(db_path, mock_embedder):
    """Vault.delete() removes stored memory."""
    v = Vault(db_path, embedder=mock_embedder)
    result = v.store("Temporary fact to be deleted soon")
    v.delete(id=result["id"])

    results = v.search("Temporary fact deleted")
    assert len(results) == 0
    v.close()


def test_vault_store_and_update(db_path, mock_embedder):
    """Vault.update() replaces content."""
    v = Vault(db_path, embedder=mock_embedder)
    result = v.store("Old content that will be replaced")
    v.update(result["id"], content="New content here now completely different")

    results = v.search("New content completely different")
    assert len(results) > 0
    v.close()


def test_vault_search_source_filter(db_path, mock_embedder):
    """Vault.search(source='memories') filters correctly."""
    v = Vault(db_path, embedder=mock_embedder)
    v.store("Memory about databases and infrastructure", namespace="facts")

    results = v.search("databases", source="memories")
    assert all(r["file_path"].startswith("memory://") for r in results)
    v.close()


def test_vault_index_wraps_directory(db_path, mock_embedder, tmp_path):
    """Vault.index() indexes markdown files from a directory."""
    p = tmp_path / "testnotes" / "note.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# Test Note\n\nSome content about testing things here with enough words for a proper chunk to be indexed."
    )
    v = Vault(db_path, embedder=mock_embedder)
    v.index(str(p.parent))

    results = v.search("testing")
    assert len(results) > 0
    v.close()
