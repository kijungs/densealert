all: compile demo
compile:
	-chmod u+x ./*.sh
	./compile.sh
demo:
	gunzip -c example_data.txt.gz > example_data.txt
	@echo [DEMO] running DenseStream...
	java -cp ./DenseAlert-1.0.jar densealert.DenseStreamExample
	@echo [DEMO] running DenseAlert...
	java -cp ./DenseAlert-1.0.jar densealert.DenseAlertExample
