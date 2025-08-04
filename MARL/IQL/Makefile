# Simplest Makefile for your C++ project

# Compiler
CXX = g++

# Compiler Flags
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pedantic -g
# -I/usr/include/eigen3 is included directly in INCLUDE_PATHS

# Directories
SRC_DIR = src
INCLUDE_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Create directories if they don't exist
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

# Include Paths (for headers)
INCLUDE_PATHS = -I$(INCLUDE_DIR) 

# Libraries to link
LDLIBS = -lm

# Source files (just the ones you specified)
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/Agent.cpp \
       $(SRC_DIR)/Environment.cpp \

# Object files (derived from source files)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Executable name
TARGET = $(BIN_DIR)/IQL

# --- Rules ---

# Default target: build the executable
all: $(TARGET)

# Rule to link object files into the executable
$(TARGET): $(OBJS) 
	@echo "Linking $(TARGET)..."
	$(CXX) $(OBJS) -o $(TARGET) $(LDLIBS)
	@echo "Build successful! Executable: $(TARGET)"

# Rule to compile .cpp files into .o object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Clean target
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJ_DIR)/*.o
	rm -f $(TARGET)
	@echo "Clean complete."

.PHONY: all clean
