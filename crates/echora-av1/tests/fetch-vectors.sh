#!/bin/sh
# Downloads more AOM conformance vectors (libaom's test/test_vectors.cc)
# for the rav1d cross-check:
#
#   crates/echora-av1/tests/fetch-vectors.sh /tmp/av1-vectors
#   ECHORA_AV1_VECTORS=/tmp/av1-vectors cargo test -p echora-av1 --release
#
# The vectors are the Alliance for Open Media's, under the BSD 2-Clause
# license and the AOM Patent License 1.0.
set -eu
dir=${1:?usage: fetch-vectors.sh DIR}
mkdir -p "$dir"
base=https://storage.googleapis.com/aom-test-data
for name in \
    av1-1-b8-02-allintra \
    av1-1-b8-16-intra_only-intrabc-extreme-dv \
    av1-1-b8-22-svc-L2T1 \
    av1-1-b8-22-svc-L2T2 \
    av1-1-b8-00-quantizer-00 \
    av1-1-b8-00-quantizer-31 \
    av1-1-b8-00-quantizer-63 \
    av1-1-b10-00-quantizer-00 \
    av1-1-b10-00-quantizer-63 \
    av1-1-b8-01-size-196x196 av1-1-b8-01-size-198x200 av1-1-b8-01-size-200x202 \
    av1-1-b8-01-size-208x210 av1-1-b8-01-size-224x226 av1-1-b8-01-size-226x196 \
    av1-1-b8-01-size-16x64 av1-1-b8-01-size-34x34 av1-1-b8-01-size-64x16 av1-1-b8-01-size-66x34
do
    [ -f "$dir/$name.ivf" ] || curl -sSfo "$dir/$name.ivf" "$base/$name.ivf"
done
