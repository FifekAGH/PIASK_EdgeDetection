CXX      := mpicc
CXXFLAGS := -O3 -qopenmp -I./src/include -I./lib/stb_image/include
SRC      := ./lib/stb_image/image.cpp ./src/main.cpp ./src/sobel.cpp 
TARGET   := sobel

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

.PHONY: clean

clean:
	rm -f $(TARGET)
