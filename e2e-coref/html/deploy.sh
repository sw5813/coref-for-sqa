#!/bin/bash

html_root=~/public_html
name=e2e-coref
mkdir -p $html_root/$name
cp index.html main.js main.css query.cgi $html_root/$name
