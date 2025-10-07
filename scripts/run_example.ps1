param(
    [string]$input,
    [string]$out = "out"
)
python -m texbank.cli --input $input --out $out
