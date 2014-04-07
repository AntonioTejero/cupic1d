# Configuration

CC = g++
NVCC = nvcc
ARCHITECTURE = sm_20
NVCCFLAGS = -arch=$(ARCHITECTURE) #-Xptxas -v
LINKERFLAGS = -arch=$(ARCHITECTURE) -lcurand 

OBJECTS = main.o init.o cc.o mesh.o particles.o diagnostic.o cuda.o


# Makefile orders

CUPIC : $(OBJECTS)
	$(NVCC) $(LINKERFLAGS) $(OBJECTS) -o cupic
	rm *~ -f
	mv ./cupic ../bin/cupic

main.o : main.cu 
	$(NVCC) $(NVCCFLAGS) -dc main.cu -o main.o 

init.o : init.cu init.h
	$(NVCC) $(NVCCFLAGS) -dc init.cu -o init.o 

cc.o : cc.cu cc.h
	$(NVCC) $(NVCCFLAGS) -dc cc.cu -o cc.o
	
mesh.o : mesh.cu mesh.h
	$(NVCC) $(NVCCFLAGS) -dc mesh.cu -o mesh.o
	
particles.o : particles.cu particles.h
	$(NVCC) $(NVCCFLAGS) -dc particles.cu -o particles.o
	
diagnostic.o : diagnostic.cu diagnostic.h
	$(NVCC) $(NVCCFLAGS) -dc diagnostic.cu -o diagnostic.o
	
cuda.o : cuda.cu cuda.h
	$(NVCC) $(NVCCFLAGS) -dc cuda.cu -o cuda.o
	
.PHONY : clean open lines

clean :
	rm *.o *~ *.dat -f
	clear

open :
	kate stdh.h main.cu init.h init.cu cc.h cc.cu mesh.h mesh.cu particles.h particles.cu diagnostic.h diagnostic.cu cuda.h cuda.cu makefile ../input/input_data &
	
lines :
	git ls-files | xargs wc -l
