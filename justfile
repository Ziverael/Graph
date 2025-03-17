[doc('List all recipes')]
default:
    just --list

[doc('Describe running PC')]
device-info:
    @echo "OS: {{os()}}"
    @echo "Family: {{os_family()}}"
    @echo "Archtecture: {{arch()}}"
    @echo "CPUs: {{num_cpus()}}"


test test_path='':
    @echo "Running tests..."
    poetry run pytest {{test_path}}

check:
    @echo "Checking code..."
    poetry run ruff check . || ERROR=$?
    poetry run ruff format --check . || ERROR=$?
    poetry run mypy --incremental --show-error-codes --pretty --check-untyped-defs . || ERROR=$?


format:
    @echo "Formatting code..."
    poetry run ruff format . || ERROR=$?
    poetry run ruff check --fix || ERROR=$?

all: check format test

alias c := check
alias f := format
alias t := test
alias a := all