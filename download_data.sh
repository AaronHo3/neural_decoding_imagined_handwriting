#!/usr/bin/env bash
#
# Download the Willett handwriting BCI dataset from Dryad.
# https://doi.org/10.5061/dryad.wh70rxwmv
#
# Dryad uses a browser-based bot challenge, so automatic download may not
# work. If curl fails, the script will open the download page in your
# browser for manual download.
#
# Usage:
#   ./download_data.sh              # extracts to current directory
#   ./download_data.sh /some/path   # extracts to specified directory

set -euo pipefail

DEST="${1:-.}"
TARBALL="$DEST/handwritingBCIData.tar.gz"
DOWNLOAD_URL="https://datadryad.org/downloads/file_stream/1099376"
LANDING_PAGE="https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwmv"

mkdir -p "$DEST"

echo "=== Willett Handwriting BCI Dataset ==="
echo ""
echo "Source:      https://doi.org/10.5061/dryad.wh70rxwmv"
echo "Destination: $(cd "$DEST" && pwd)"
echo "Size:        ~1.4 GB download"
echo ""

# Check if already extracted
if [ -d "$DEST/handwritingBCIData/Datasets" ]; then
    echo "Dataset already exists at $DEST/handwritingBCIData/"
    echo "Delete it first if you want to re-download."
    exit 0
fi

# Check if tarball already exists (e.g. from manual browser download)
if [ -f "$TARBALL" ]; then
    echo "Tarball found at $TARBALL"
else
    # Also check common browser download locations
    FOUND=""
    for candidate in \
        "$HOME/Downloads/handwritingBCIData.tar.gz" \
        "$HOME/Downloads/doi_10.5061_dryad.wh70rxwmv.zip" \
        "$HOME/Desktop/handwritingBCIData.tar.gz"; do
        if [ -f "$candidate" ]; then
            FOUND="$candidate"
            break
        fi
    done

    if [ -n "$FOUND" ]; then
        echo "Found download at: $FOUND"
        echo "Moving to $TARBALL"
        cp "$FOUND" "$TARBALL"
    else
        echo "Tarball not found. Attempting download..."
        echo ""

        # Try curl first (may fail due to Dryad's JS bot challenge)
        if curl -L -o "$TARBALL" --progress-bar --fail "$DOWNLOAD_URL" 2>/dev/null; then
            # Verify it's actually a tar.gz and not an HTML error page
            if file "$TARBALL" | grep -q "gzip"; then
                echo "Download complete."
            else
                echo ""
                echo "Download returned an HTML page (Dryad's bot protection)."
                rm -f "$TARBALL"
            fi
        else
            rm -f "$TARBALL"
        fi

        if [ ! -f "$TARBALL" ]; then
            echo "============================================================"
            echo " Automatic download blocked by Dryad's bot protection."
            echo ""
            echo " Please download manually:"
            echo ""
            echo "   1. Open: $LANDING_PAGE"
            echo "   2. Click 'Download dataset' (handwritingBCIData.tar.gz)"
            echo "   3. Save to: $(cd "$DEST" && pwd)/"
            echo ""
            echo " Then re-run this script."
            echo "============================================================"

            # Try to open the page in the browser
            if command -v open &>/dev/null; then
                open "$LANDING_PAGE"
            elif command -v xdg-open &>/dev/null; then
                xdg-open "$LANDING_PAGE"
            fi
            exit 1
        fi
    fi
fi

# Detect format and extract
echo "Extracting..."
if file "$TARBALL" | grep -q "gzip"; then
    tar -xzf "$TARBALL" -C "$DEST"
elif file "$TARBALL" | grep -q "Zip"; then
    unzip -q "$TARBALL" -d "$DEST"
else
    echo "Error: Unrecognised archive format."
    file "$TARBALL"
    exit 1
fi
echo ""

# Verify structure
if [ -d "$DEST/handwritingBCIData/Datasets" ]; then
    echo "=== Download successful ==="
    echo ""
    echo "Sessions:"
    ls -1 "$DEST/handwritingBCIData/Datasets/"
    echo ""
    echo "Pre-computed steps:"
    ls -1 "$DEST/handwritingBCIData/RNNTrainingSteps/" 2>/dev/null || echo "  (none)"
    echo ""
    echo "To use in Python:"
    echo "  from data import WillettDataset"
    echo "  ds = WillettDataset('$(cd "$DEST" && pwd)/handwritingBCIData')"
else
    echo "Warning: Expected directory structure not found."
    echo "Contents of $DEST:"
    ls -1 "$DEST"
fi

echo ""
echo "Tarball kept at $TARBALL (delete manually to free ~1.4 GB)."
