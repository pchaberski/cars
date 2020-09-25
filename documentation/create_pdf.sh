pandoc \
	--toc \
	-V geometry:margin=1.2in \
	-V fontsize=12pt \
	-V fontfamily:fourier \
	--top-level-division=chapter \
	1_introduction.md 2_project_description.md 3_experimentation_setup.md 4_results.md \
	-o thesis.pdf