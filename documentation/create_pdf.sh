cat 1_introduction.md 2_project_description.md 3_experimentation_setup.md 4_results.md 5_references.md >> merged.md
sed -e s/5_references.md//g -i merged.md
pandoc --toc -H options.sty -V geometry:margin=1.2in -V fontsize=12pt -V fontfamily:fourier --top-level-division=chapter merged.md -o thesis.pdf
rm merged.md
