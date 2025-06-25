#!/usr/bin/env bash


for i in {0..530}; do
    mysql --user=etu0623 --password="$1" --database=etu0623 < ./points_final_"$i".sql
    echo "Done points_final_${i}.sql"
done
