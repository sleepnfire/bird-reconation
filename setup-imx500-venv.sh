#!/bin/bash
# Crée un venv Python 3.12 dédié à l'export IMX500 (Sony MCT).
#
# MCT exige matplotlib<3.10, incompatible avec Python 3.14.
# Ce venv séparé sert uniquement pour la commande :
#   .venv-imx500/bin/python export.py --checkpoint output/best_mobilenetv2.pth --target imx500
#
# Pré-requis : Python 3.12 installé (python3.12)

set -e

VENV_DIR=".venv-imx500"

if [ -d "$VENV_DIR" ]; then
    echo "Le venv $VENV_DIR existe déjà."
    echo "Pour le recréer : rm -rf $VENV_DIR && $0"
    exit 1
fi

PYTHON_BIN=""
for candidate in python3.12 python3.11; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3.11 ou 3.12 requis. Installable via :"
    echo "  macOS   : brew install python@3.12"
    echo "  Ubuntu  : sudo apt install python3.12 python3.12-venv"
    echo "  Windows : winget install Python.Python.3.12"
    exit 1
fi

echo "Création du venv avec $PYTHON_BIN..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "Installation de PyTorch..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install torch torchvision

echo "Installation des dépendances IMX500..."
"$VENV_DIR/bin/pip" install -r requirements-imx500.txt

echo ""
echo "=== Venv IMX500 prêt ==="
echo ""
echo "Usage :"
echo "  $VENV_DIR/bin/python export.py --checkpoint output/best_mobilenetv2.pth --target imx500"
echo "  $VENV_DIR/bin/python export.py --checkpoint output/best_mobilenetv2_distilled.pth --target imx500"
echo ""
echo "Tests :"
echo "  $VENV_DIR/bin/python -m pytest tests/test_export.py::TestExportIMX500 -v"
