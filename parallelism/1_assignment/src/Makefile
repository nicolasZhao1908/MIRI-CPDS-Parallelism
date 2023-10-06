MPICC    = mpicc

#DEFS     = 
#BASECFLAGS   = $(DEFS) -O3 -std=c99 -march=native 
BASECFLAGS   = $(DEFS) -O3 -march=native 
CFLAGS   = $(BASECFLAGS) -Wall -Wimplicit-function-declaration 
CFLAGS   = $(BASECFLAGS)

LFLAGS = -lm

TARGETS = par_data_struct par_data_struct_nonblocking par_data_struct_sendreceive Jacobi Jacobi_nb Jacobi_vr

all: $(TARGETS)

par_data_struct: par_data_struct.c
	$(MPICC) $< $(CFLAGS) $(CINCL) -o $@ $(CLIBS)

par_data_struct_nonblocking: par_data_struct_nonblocking.c
	$(MPICC) $< $(CFLAGS) $(CINCL) -o $@ $(CLIBS)

par_data_struct_sendreceive: par_data_struct_sendreceive.c
	$(MPICC) $< $(CFLAGS) $(CINCL) -o $@ $(CLIBS)

Jacobi: Jacobi.c
	$(MPICC) $< $(CFLAGS) $(CINCL) -o $@ $(CLIBS) $(LFLAGS)

Jacobi_nb: Jacobi_nb.c
	$(MPICC) $< $(CFLAGS) $(CINCL) -o $@ $(CLIBS) $(LFLAGS)

Jacobi_vr: Jacobi_vr.c
	$(MPICC) $< $(CFLAGS) $(CINCL) -o $@ $(CLIBS) $(LFLAGS)

clean:
	rm -rf $(TARGETS) *.o *.o* *.e*

ultraclean:
	rm -rf $(TARGETS) *.o *.o* *.e* Output*.txt *.ps *.prv *.pcf *.row

