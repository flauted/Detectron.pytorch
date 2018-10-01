# builds the target package.

echo "STARTING PKG BUILD."
DIR="$(dirname "$(readlink -f "$0")")"
echo "DIR: ${DIR}"
cd "${DIR}/mask-rcnn.pytorch/lib"
bash make.sh
