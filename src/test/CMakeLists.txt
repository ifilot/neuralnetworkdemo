# ###
# add individual tests
# ###

#######################################################
# network test
#######################################################
add_executable(TestNeuralNetwork
               unittest.cpp
               neuralnetworktest.cpp
               ../neural_network.cpp
              )
target_link_libraries(TestNeuralNetwork cppunit openblas)

#######################################################
# add tests to the set
#######################################################
add_test(NAME "TestNeuralNetwork" COMMAND TestNeuralNetwork)
