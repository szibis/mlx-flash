# One-command local MLX-Flash runtime. Inference runs natively for Metal;
# Docker Compose manages local monitoring and test services.

SHELL := /bin/bash
BASE_PYTHON ?= $(shell command -v python3.13 || command -v python3.12 || command -v python3)
PYTHON ?= .venv/bin/python
BASE_PYTHON_VERSION := $(shell "$(BASE_PYTHON)" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)
VISION ?= 0
ifeq ($(VISION),1)
MODEL ?= mlx-community/gemma-4-e4b-it-4bit
PORT ?= 8082
else
MODEL ?= mlx-community/Qwen3-4B-Instruct-2507-4bit
PORT ?= 8080
endif
HOST ?= 127.0.0.1

export MLX_FLASH_MODEL := $(MODEL)
export MLX_FLASH_HOST := $(HOST)
export MLX_FLASH_PORT := $(PORT)
export MLX_FLASH_VISION := $(VISION)

.PHONY: help setup up vision-up status verify verify-vision logs down test test-native test-e2e test-rust docker-ready clean

help:
	@echo "MLX-Flash — proste polecenia"
	@echo "  make up       przygotuj środowisko i uruchom całość"
	@echo "  make vision-up uruchom lokalny model obrazu na porcie 8082"
	@echo "  make status   pokaż stan serwera i monitoringu"
	@echo "  make verify   sprawdź działające API i model"
	@echo "  make verify-vision sprawdź model obrazu syntetycznym zdjęciem"
	@echo "  make logs     pokaż bieżące logi serwera"
	@echo "  make down     bezpiecznie zatrzymaj serwer i kontenery"
	@echo "  make test     uruchom testy w Dockerze"
	@echo ""
	@echo "Model można zmienić: make up MODEL=mlx-community/inna-nazwa"
	@echo "Vision: make vision-up (Gemma 4 E4B, port 8082)"

setup:
	@python3 scripts/local_runtime.py check-host
	@if [ ! -x "$(BASE_PYTHON)" ]; then echo "Nie znaleziono Python 3.12/3.13. Zainstaluj Python 3.13 i ponów próbę."; exit 1; fi
	@if [ ! -x "$(PYTHON)" ] || [ "$$($(PYTHON) -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)" != "$(BASE_PYTHON_VERSION)" ]; then \
		echo "Przygotowuję środowisko dla Python $(BASE_PYTHON_VERSION)…"; \
		rm -rf .venv; "$(BASE_PYTHON)" -m venv .venv || exit $$?; \
	fi
	@if ! "$(PYTHON)" -m pip --version >/dev/null 2>&1; then \
		echo "Uzupełniam pip w lokalnym środowisku…"; "$(PYTHON)" -m ensurepip --upgrade || exit $$?; \
	fi
	@if [ ! -f .venv/.mlxflash-ready ]; then \
		echo "Przygotowuję lokalny silnik MLX i wymagane pakiety…"; \
		"$(PYTHON)" -m pip install --disable-pip-version-check -e '.[all]' || exit $$?; \
		touch .venv/.mlxflash-ready; \
	else echo "Lokalne środowisko MLX-Flash jest już przygotowane."; fi
	@if [ "$(VISION)" = "1" ] && [ ! -f .venv/.mlxflash-vision-ready ]; then \
		"$(PYTHON)" -m pip install --disable-pip-version-check -e '.[vision]' || exit $$?; \
		touch .venv/.mlxflash-vision-ready; \
	fi
	@echo "Gotowe."

up: docker-ready setup
	@"$(PYTHON)" scripts/local_runtime.py up

vision-up:
	@$(MAKE) up VISION=1 PORT=8082 MODEL=mlx-community/gemma-4-e4b-it-4bit

status:
	@python3 scripts/local_runtime.py status

verify:
	@"$(PYTHON)" scripts/verify_openai_api.py --base-url "http://$(HOST):$(PORT)"

verify-vision:
	@"$(PYTHON)" scripts/verify_vision_api.py --base "http://$(HOST):$(PORT)" --out .local/vision-api-proof.json

logs:
	@mkdir -p .local/runtime
	@tail -n 80 -f .local/runtime/server.log

down:
	@python3 scripts/local_runtime.py down

test: docker-ready
	@docker compose run --build --rm tests

test-native:
	@"$(PYTHON)" -m pytest tests/ -v --tb=short

test-e2e: docker-ready
	@docker compose run --build --rm e2e

test-rust: docker-ready
	@docker compose run --build --rm rust-build

docker-ready:
	@docker info >/dev/null 2>&1 || { echo "Docker Desktop nie jest uruchomiony. Otwórz Docker Desktop i poczekaj, aż będzie gotowy."; exit 1; }

clean:
	@python3 scripts/local_runtime.py down
	@rm -rf .local .venv
	@echo "Lokalne środowisko MLX-Flash usunięte. Pobrane modele w cache pozostają na dysku."
