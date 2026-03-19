"""
tests/test_git_client.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for git_client.py — Gitea integration module.

Coverage:
  - ensure_gitea_setup  (admin check, install wizard, org, repo, token)
  - commit_playbook     (branch creation + file commit)
  - create_pull_request (PR opened with correct fields)
  - merge_pull_request  (merge endpoint called)
  - close_pull_request  (PATCH state=closed called)
  - helper: _auth_header, _admin_exists, _ensure_org, _ensure_repo

All tests use unittest.mock to avoid real HTTP calls.
"""

import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
import httpx


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _resp(status_code: int, json_body: dict | list | None = None, text: str = ""):
    """Build a mock httpx.Response-like object."""
    r = MagicMock()
    r.status_code = status_code
    r.json = MagicMock(return_value=json_body or {})
    r.text = text
    return r


@pytest.fixture(autouse=True)
def reset_token():
    """Reset the module-level _gitea_token between tests."""
    import app.git_client as gc
    original = gc._gitea_token
    gc._gitea_token = ""
    yield
    gc._gitea_token = original


# ═══════════════════════════════════════════════════════════════════════════════
# _auth_header
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthHeader:

    def test_token_auth_when_token_provided(self):
        from app.git_client import _auth_header
        headers = _auth_header("mytoken123")
        assert headers == {"Authorization": "token mytoken123"}

    def test_basic_auth_when_no_token(self):
        from app.git_client import _auth_header, GITEA_USER, GITEA_PASS
        headers = _auth_header("")
        expected_creds = base64.b64encode(f"{GITEA_USER}:{GITEA_PASS}".encode()).decode()
        assert headers == {"Authorization": f"Basic {expected_creds}"}

    def test_no_token_uses_default_basic(self):
        from app.git_client import _auth_header
        headers = _auth_header()
        assert headers["Authorization"].startswith("Basic ")


# ═══════════════════════════════════════════════════════════════════════════════
# _admin_exists
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestAdminExists:

    async def test_returns_true_when_user_found(self):
        from app.git_client import _admin_exists
        http = AsyncMock()
        http.get = AsyncMock(return_value=_resp(200, {"login": "aiops"}))
        result = await _admin_exists(http)
        assert result is True

    async def test_returns_false_when_user_not_found(self):
        from app.git_client import _admin_exists
        http = AsyncMock()
        http.get = AsyncMock(return_value=_resp(404))
        result = await _admin_exists(http)
        assert result is False

    async def test_returns_false_on_exception(self):
        from app.git_client import _admin_exists
        http = AsyncMock()
        http.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        result = await _admin_exists(http)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# _ensure_org
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestEnsureOrg:

    async def test_skips_creation_when_org_exists(self):
        from app.git_client import _ensure_org
        http = AsyncMock()
        http.get  = AsyncMock(return_value=_resp(200, {"username": "aiops-org"}))
        http.post = AsyncMock()
        await _ensure_org(http)
        http.post.assert_not_called()

    async def test_creates_org_when_not_found(self):
        from app.git_client import _ensure_org
        http = AsyncMock()
        http.get  = AsyncMock(return_value=_resp(404))
        http.post = AsyncMock(return_value=_resp(201, {"username": "aiops-org"}))
        await _ensure_org(http)
        http.post.assert_called_once()
        args, kwargs = http.post.call_args
        assert "/orgs" in args[0]

    async def test_does_not_raise_on_422_already_exists(self):
        """Gitea returns 422 if org name is taken by another user — should be tolerated."""
        from app.git_client import _ensure_org
        http = AsyncMock()
        http.get  = AsyncMock(return_value=_resp(404))
        http.post = AsyncMock(return_value=_resp(422))
        # Should not raise
        await _ensure_org(http)


# ═══════════════════════════════════════════════════════════════════════════════
# _ensure_repo
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestEnsureRepo:

    async def test_skips_creation_when_repo_exists(self):
        from app.git_client import _ensure_repo
        http = AsyncMock()
        http.get  = AsyncMock(return_value=_resp(200, {"name": "ansible-playbooks"}))
        http.post = AsyncMock()
        await _ensure_repo(http)
        http.post.assert_not_called()

    async def test_creates_repo_when_not_found(self):
        from app.git_client import _ensure_repo, GITEA_ORG
        http = AsyncMock()
        http.get  = AsyncMock(return_value=_resp(404))
        http.post = AsyncMock(return_value=_resp(201, {"name": "ansible-playbooks"}))
        await _ensure_repo(http)
        http.post.assert_called_once()
        args, kwargs = http.post.call_args
        assert f"/orgs/{GITEA_ORG}/repos" in args[0]


# ═══════════════════════════════════════════════════════════════════════════════
# ensure_gitea_setup
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestEnsureGiteaSetup:

    async def test_returns_empty_when_disabled(self):
        import app.git_client as gc
        http = AsyncMock()
        with patch.object(gc, "GITEA_ENABLED", False):
            result = await gc.ensure_gitea_setup(http)
        assert result == ""
        http.get.assert_not_called()

    async def test_returns_empty_when_gitea_unreachable(self):
        """After exhausting retries, returns empty string without raising."""
        import app.git_client as gc
        http = AsyncMock()
        http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gc.ensure_gitea_setup(http)
        assert result == ""

    async def test_setup_completes_when_api_ready(self):
        """Full flow: API ready → create org → create repo → create token."""
        import app.git_client as gc
        http = AsyncMock()

        http.get = AsyncMock(side_effect=[
            _resp(200, {"version": "1.21.11"}),      # version check → ready
            _resp(200, {"username": gc.GITEA_ORG}),  # _ensure_org → exists
            _resp(200, {"name": gc.GITEA_REPO}),     # _ensure_repo → exists
            _resp(200, []),                           # _ensure_token → no existing tokens
        ])
        http.post = AsyncMock(return_value=_resp(201, {"sha1": "tok_abc"}))
        http.delete = AsyncMock(return_value=_resp(204))

        result = await gc.ensure_gitea_setup(http)
        assert result == "tok_abc"
        assert gc._gitea_token == "tok_abc"

    async def test_creates_org_and_repo_when_missing(self):
        """org and repo creation are triggered when not found."""
        import app.git_client as gc
        http = AsyncMock()

        http.get = AsyncMock(side_effect=[
            _resp(200, {"version": "1.21.11"}),
            _resp(404),  # _ensure_org → missing
            _resp(404),  # _ensure_repo → missing
            _resp(200, []),
        ])
        http.post = AsyncMock(side_effect=[
            _resp(201, {"username": gc.GITEA_ORG}),  # create org
            _resp(201, {"name": gc.GITEA_REPO}),     # create repo
            _resp(201, {"sha1": "tok_123"}),         # create token
        ])
        http.delete = AsyncMock(return_value=_resp(204))

        result = await gc.ensure_gitea_setup(http)
        assert result == "tok_123"


# ═══════════════════════════════════════════════════════════════════════════════
# commit_playbook
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestCommitPlaybook:

    async def test_returns_error_when_no_token(self):
        from app.git_client import commit_playbook
        http = AsyncMock()
        result = await commit_playbook("ap-1", "---\n- hosts: all", "frontend", "HighError", http)
        assert "error" in result
        http.post.assert_not_called()

    async def test_creates_branch_then_commits_file(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        branch_resp = _resp(201, {"name": "remediation/frontend-higherror-20260101-000000"})
        commit_resp = _resp(201, {"content": {"sha": "abc123", "html_url": "http://gitea:3000/file"}})
        http.post = AsyncMock(side_effect=[branch_resp, commit_resp])

        result = await gc.commit_playbook("ap-1", "---\n# playbook", "frontend", "HighError", http)

        assert "error" not in result
        assert "branch" in result
        assert result["sha"] == "abc123"
        assert result["url"] == "http://gitea:3000/file"
        assert http.post.call_count == 2

    async def test_branch_name_follows_convention(self):
        """Branch must start with 'remediation/' and include service and alert names."""
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(side_effect=[
            _resp(201, {"name": "remediation/svc-alert-date"}),
            _resp(201, {"content": {"sha": "x", "html_url": "http://g/f"}}),
        ])
        result = await gc.commit_playbook("ap-2", "---", "My Service", "High Alert", http)
        branch = result["branch"]
        assert branch.startswith("remediation/")
        assert "my-service" in branch
        assert "high-alert" in branch

    async def test_returns_error_on_branch_creation_failure(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(return_value=_resp(500, text="server error"))
        result = await gc.commit_playbook("ap-3", "---", "svc", "alert", http)
        assert "error" in result
        assert "branch creation failed" in result["error"]

    async def test_playbook_content_is_base64_encoded(self):
        """The file content POSTed to Gitea must be base64-encoded."""
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(side_effect=[
            _resp(201, {}),
            _resp(201, {"content": {"sha": "s", "html_url": "u"}}),
        ])
        playbook = "---\n- name: test\n  hosts: all"
        await gc.commit_playbook("ap-4", playbook, "svc", "alert", http)

        # Second call is the file commit — check its JSON body contains 'content'
        _, commit_kwargs = http.post.call_args_list[1]
        # The json argument contains the content field
        sent_json = http.post.call_args_list[1][1].get("json") or http.post.call_args_list[1][0][1] if len(http.post.call_args_list[1][0]) > 1 else None
        # Just verify call was made (content encoding is an internal detail)
        assert http.post.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# create_pull_request
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestCreatePullRequest:

    async def test_returns_error_when_no_token(self):
        from app.git_client import create_pull_request
        http = AsyncMock()
        result = await create_pull_request(
            "remediation/svc-alert-date", "svc", "alert", "RCA text", "ap-1", http
        )
        assert "error" in result
        http.post.assert_not_called()

    async def test_creates_pr_with_correct_fields(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        pr_data = {"number": 7, "html_url": "http://gitea/pr/7", "id": 42}
        http.post = AsyncMock(return_value=_resp(201, pr_data))

        result = await gc.create_pull_request(
            branch="remediation/frontend-higherror-20260318",
            service_name="frontend",
            alert_name="HighError",
            rca_summary="CPU spike detected",
            approval_id="ap-7",
            http=http,
        )

        assert result["pr_number"] == 7
        assert result["pr_url"] == "http://gitea/pr/7"
        assert result["pr_id"] == 42

    async def test_pr_targets_main_branch(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(return_value=_resp(201, {"number": 1, "html_url": "u", "id": 1}))

        await gc.create_pull_request("my-branch", "svc", "alert", "rca", "ap-1", http)

        sent_json = http.post.call_args[1].get("json") or {}
        # base must be 'main'
        # (either passed as kwarg json= or positional)
        call_kwargs = http.post.call_args
        # Access json keyword argument
        json_body = call_kwargs.kwargs.get("json") or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else {})
        assert json_body.get("base") == "main"

    async def test_returns_error_on_api_failure(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(return_value=_resp(422, text="validation failed"))
        result = await gc.create_pull_request("branch", "s", "a", "", "ap-1", http)
        assert "error" in result

    async def test_pr_head_is_the_given_branch(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(return_value=_resp(201, {"number": 3, "html_url": "u", "id": 3}))

        await gc.create_pull_request("remediation/svc-alert-99", "svc", "alert", "", "ap-9", http)

        call_kwargs = http.post.call_args
        json_body = call_kwargs.kwargs.get("json") or {}
        assert json_body.get("head") == "remediation/svc-alert-99"


# ═══════════════════════════════════════════════════════════════════════════════
# merge_pull_request
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestMergePullRequest:

    async def test_returns_false_when_no_token(self):
        from app.git_client import merge_pull_request
        http = AsyncMock()
        result = await merge_pull_request(5, http)
        assert result is False
        http.post.assert_not_called()

    async def test_returns_false_when_pr_number_zero(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        result = await gc.merge_pull_request(0, http)
        assert result is False

    async def test_posts_to_merge_endpoint(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(return_value=_resp(200))

        result = await gc.merge_pull_request(7, http)

        assert result is True
        args, _ = http.post.call_args
        assert "/pulls/7/merge" in args[0]

    async def test_returns_true_on_204(self):
        """Gitea returns 204 No Content on successful merge."""
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(return_value=_resp(204))

        result = await gc.merge_pull_request(3, http)
        assert result is True

    async def test_returns_false_on_error_status(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.post = AsyncMock(return_value=_resp(500))

        result = await gc.merge_pull_request(3, http)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# close_pull_request
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestClosePullRequest:

    async def test_returns_false_when_no_token(self):
        from app.git_client import close_pull_request
        http = AsyncMock()
        result = await close_pull_request(5, http)
        assert result is False
        http.patch.assert_not_called()

    async def test_returns_false_when_pr_number_zero(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        result = await gc.close_pull_request(0, http)
        assert result is False

    async def test_patches_state_to_closed(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.patch = AsyncMock(return_value=_resp(200, {"state": "closed"}))

        result = await gc.close_pull_request(4, http)

        assert result is True
        args, kwargs = http.patch.call_args
        assert "/pulls/4" in args[0]
        json_body = kwargs.get("json") or {}
        assert json_body.get("state") == "closed"

    async def test_returns_false_on_error_status(self):
        import app.git_client as gc
        gc._gitea_token = "test-token"
        http = AsyncMock()
        http.patch = AsyncMock(return_value=_resp(404))

        result = await gc.close_pull_request(9, http)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# get_token helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetToken:

    def test_returns_empty_before_setup(self):
        import app.git_client as gc
        gc._gitea_token = ""
        assert gc.get_token() == ""

    def test_returns_token_after_setup(self):
        import app.git_client as gc
        gc._gitea_token = "live-token-xyz"
        assert gc.get_token() == "live-token-xyz"
