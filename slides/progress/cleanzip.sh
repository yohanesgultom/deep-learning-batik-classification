#!/bin/sh

rm -f *.*~
rm -f *.aux
rm -f *.snm
rm -f *.blg
rm -f *.bbl
rm -f *.nav
rm -f *.toc
rm -f *.aux
rm -f *.log
rm -f *.out
rm -f *.synctex.gz

# archive
zip_name=batik_classification_deeplearning_yohanesgultom
rm -f $zip_name.zip && 
zip $zip_name.zip *.*
