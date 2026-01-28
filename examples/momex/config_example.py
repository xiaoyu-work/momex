"""Momex Configuration Examples.

This example demonstrates how to configure Momex using both code and YAML.

Prerequisites:
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_MODEL=gpt-4o

    # For PostgreSQL:
    pip install momex[postgres]
"""

import asyncio
from momex import Memory, MomexConfig, StorageConfig, PostgresConfig


async def example_sqlite_code():
    """Configure SQLite backend using code."""
    print("=== SQLite Configuration (Code) ===")

    # Method 1: Simple - use defaults
    memory = Memory(collection="user:xiaoyuzhang")
    print(f"Default path: {memory.db_path}")

    # Method 2: Custom storage path
    config = MomexConfig(
        storage=StorageConfig(path="./my_custom_data")
    )
    memory = Memory(collection="user:xiaoyuzhang", config=config)
    print(f"Custom path: {memory.db_path}")

    # Method 3: Set global default
    MomexConfig.set_default(storage_path="./global_data")
    memory = Memory(collection="user:xiaoyuzhang")  # Uses global default
    print(f"Global default path: {memory.db_path}")
    MomexConfig.clear_default()


async def example_postgres_code():
    """Configure PostgreSQL backend using code."""
    print("\n=== PostgreSQL Configuration (Code) ===")

    config = MomexConfig(
        backend="postgres",
        postgres=PostgresConfig(
            url="postgresql://user:password@localhost:5432/momex",
            pool_min=2,
            pool_max=10,
        )
    )

    print(f"Backend: {config.backend}")
    print(f"PostgreSQL URL: {config.postgres.url}")
    print(f"Pool size: {config.postgres.pool_min}-{config.postgres.pool_max}")

    # Note: Memory will connect when first used
    # memory = Memory(collection="user:xiaoyuzhang", config=config)


async def example_yaml_config():
    """Configure using YAML files."""
    print("\n=== YAML Configuration ===")

    # Load SQLite config from YAML
    sqlite_config = MomexConfig.from_yaml("config_sqlite.yaml")
    print(f"SQLite backend: {sqlite_config.backend}")
    print(f"SQLite path: {sqlite_config.storage.path}")

    # Load PostgreSQL config from YAML
    try:
        pg_config = MomexConfig.from_yaml("config_postgres.yaml")
        print(f"PostgreSQL backend: {pg_config.backend}")
        print(f"PostgreSQL URL: {pg_config.postgres.url}")
    except Exception as e:
        print(f"PostgreSQL config: {e}")


async def example_env_vars():
    """Configure using environment variables."""
    print("\n=== Environment Variables ===")

    print("Available environment variables:")
    print("  MOMEX_BACKEND=sqlite|postgres")
    print("  MOMEX_STORAGE_PATH=./path/to/data")
    print("  MOMEX_POSTGRES_URL=postgresql://user:pass@host:5432/db")


async def example_save_config():
    """Save configuration to YAML."""
    print("\n=== Save Configuration ===")

    # Create config and save
    config = MomexConfig(
        backend="postgres",
        postgres=PostgresConfig(
            url="postgresql://user:password@localhost:5432/momex",
            pool_min=5,
            pool_max=20,
        )
    )

    # Save to file
    config.to_yaml("my_config.yaml")
    print("Config saved to my_config.yaml")

    # Verify by loading
    loaded = MomexConfig.from_yaml("my_config.yaml")
    print(f"Loaded backend: {loaded.backend}")

    # Cleanup
    import os
    os.remove("my_config.yaml")


async def main():
    await example_sqlite_code()
    await example_postgres_code()
    await example_yaml_config()
    await example_env_vars()
    await example_save_config()


if __name__ == "__main__":
    asyncio.run(main())
