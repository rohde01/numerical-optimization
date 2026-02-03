### Install Homebrew (for Emil & Mathilde)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install uv (for Emil & Mathilde)

```bash
brew install uv
```

uv is a fast Python package installer and resolver, written in Rust. It is used for reproducibility because it ensures consistent dependency versions across environments by locking dependencies in a uv.lock file.

### Clone the Repository

```bash
git clone git@github.com:rohde01/numerical-optimization.git
```

### Sync Dependencies

```bash
uv sync
```