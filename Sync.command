#!/bin/zsh
cd -- "${0:A:h}" || exit 1
/usr/bin/python3 scripts/sync.py
result=$?
printf "\nPress Return to close."
read -r _
exit "$result"
