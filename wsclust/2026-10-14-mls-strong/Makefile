# Auto-discover all step scripts (NN-name.sh pattern)
SCRIPTS := $(sort $(wildcard [0-9][0-9]-*.sh))
STEPS := $(basename $(SCRIPTS))
TARGETS := $(addprefix resultdir/,$(STEPS))

.PHONY: all clean list

# Default: run all steps (last target)
all: $(lastword $(TARGETS))

# List available steps
list:
	@echo "Scripts: $(SCRIPTS)"
	@echo "Steps:   $(STEPS)"
	@echo "Targets: $(TARGETS)"

# Clean all outputs
clean:
	rm -rf resultdir workdir

# Step rules - each depends on its script and previous resultdir
# Add new steps by following the pattern below

resultdir/00-setup: 00-setup.sh
	./00-setup.sh

resultdir/01-compile: 01-compile.sh resultdir/00-setup
	./01-compile.sh

resultdir/02-run: 02-run.sh resultdir/01-compile
	./02-run.sh

resultdir/03-build-phylo: 03-build-phylo.sh resultdir/02-run
	./03-build-phylo.sh

resultdir/04-export-phylo: 04-export-phylo.sh resultdir/03-build-phylo
	./04-export-phylo.sh

resultdir/99-tarball: 99-tarball.sh resultdir/04-export-phylo
	./99-tarball.sh
