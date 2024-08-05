#/usr/bin/bash
python -m vineyard \
    --socket '/tmp/vineyard.sock' \
    --size '32G' \
    --spill_path '/data00/tiger'



# all configs are in src/server/util/spec_resolver.cc