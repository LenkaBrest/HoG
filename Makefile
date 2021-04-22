all: hog_test
hog_test: hog_test.cpp

	g++ hog_test.cpp -o hog_test `pkg-config --cflags --libs opencv`

.PHONY: clean
clean:
	rm hog_test
