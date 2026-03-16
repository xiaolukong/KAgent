"""Tests for KAgent configuration and .env file loading."""

import os

from kagent.common import config as config_module
from kagent.common.config import KAgentConfig, configure, get_config


class TestKAgentConfig:
    """Test KAgentConfig reads from env vars and .env files."""

    def setup_method(self):
        """Reset the module-level singleton before each test."""
        config_module._config = None

    def teardown_method(self):
        """Clean up singleton and env vars after each test."""
        config_module._config = None
        os.environ.pop("KAGENT_API_KEY", None)
        os.environ.pop("KAGENT_BASE_URL", None)

    def test_default_config_has_none_values(self):
        """Without env vars or .env, config values are None."""
        cfg = KAgentConfig(_env_file=None)
        assert cfg.api_key is None
        assert cfg.base_url is None

    def test_configure_sets_values(self):
        """Explicit configure() call sets api_key and base_url."""
        cfg = configure(api_key="sk-test", base_url="https://example.com")
        assert cfg.api_key == "sk-test"
        assert cfg.base_url == "https://example.com"

    def test_get_config_returns_singleton(self):
        """get_config() returns the same instance on repeated calls."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_configure_replaces_singleton(self):
        """configure() replaces the singleton."""
        cfg1 = configure(api_key="first")
        cfg2 = configure(api_key="second")
        assert cfg1.api_key == "first"
        assert cfg2.api_key == "second"
        assert get_config().api_key == "second"

    def test_env_vars_are_picked_up(self):
        """KAgentConfig reads KAGENT_* environment variables."""
        os.environ["KAGENT_API_KEY"] = "env-key-123"
        os.environ["KAGENT_BASE_URL"] = "https://env-url.com"

        cfg = KAgentConfig(_env_file=None)
        assert cfg.api_key == "env-key-123"
        assert cfg.base_url == "https://env-url.com"

    def test_explicit_values_override_env_vars(self):
        """Explicit configure() values take precedence over env vars."""
        os.environ["KAGENT_API_KEY"] = "env-key"
        cfg = configure(api_key="explicit-key")
        assert cfg.api_key == "explicit-key"

    def test_dotenv_file_loading(self, tmp_path):
        """KAgentConfig loads values from a .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "KAGENT_API_KEY=dotenv-key-456\nKAGENT_BASE_URL=https://dotenv-url.com\n"
        )

        cfg = KAgentConfig(_env_file=str(env_file))
        assert cfg.api_key == "dotenv-key-456"
        assert cfg.base_url == "https://dotenv-url.com"

    def test_env_var_overrides_dotenv(self, tmp_path):
        """Environment variables take precedence over .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("KAGENT_API_KEY=from-dotenv\n")

        os.environ["KAGENT_API_KEY"] = "from-env-var"

        cfg = KAgentConfig(_env_file=str(env_file))
        assert cfg.api_key == "from-env-var"

    def test_configure_with_no_args_reads_env(self):
        """configure() with no args still picks up env vars."""
        os.environ["KAGENT_API_KEY"] = "auto-key"
        cfg = configure()
        assert cfg.api_key == "auto-key"
