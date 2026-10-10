#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo "Uruchamiam MLX-Flash. Przy pierwszym starcie przygotuję potrzebne pliki i model."
if make up; then
  echo "Gotowe. Lokalne API http://127.0.0.1:8080/v1 jest dostępne dla Twojej aplikacji."
else
  echo "Nie udało się uruchomić MLX-Flash. Przeczytaj komunikat powyżej."
  read -r -p "Naciśnij Enter, aby zamknąć to okno."
  exit 1
fi
