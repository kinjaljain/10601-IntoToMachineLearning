# import matplotlib.pyplot as plt
#
# plt.plot([0, 1, 2, 3, 4, 5, 6, 7], [0.4430, 0.2013, 0.1342, 0.1141, 0.1074, 0.0872, 0.0805, 0.0738], label='Training Error')
# plt.plot([0, 1, 2, 3, 4, 5, 6, 7], [0.5060, 0.2169, 0.1566, 0.1687, 0.2048, 0.1687, 0.2530, 0.2651], label='Testing Error')
#
# plt.xlabel('Depth')
# plt.ylabel('Error')
# plt.title("Politicians Data Plot")
# plt.axis([0, 7, 0, 0.52])
# plt.legend()
# plt.show()

import matplotlib.pyplot as plt

# plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [3.771, 24.016, 24.456, 24.713, 24.764, 24.930, 25.019, 24.965, 24.985, 25.003])
# # plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], [24.016, 24.456, 0.1566, 0.1687, 0.2048, 0.1687, 0.2530, , 25.003], label='Model 2')
#
# plt.xlabel('Training Sentences (in Millions)')
# plt.ylabel('BLEU Score')
# plt.title("Training Size vs Performance")
# plt.axis([0, 9.5, 23.5, 25.5])
# plt.legend()
# plt.show()

# plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [120, 200, 473, 682, 768, 840, 896, 953, 1008, 1126])
# # plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], [246, 0.2169, 0.1566, 0.1687, 0.2048, 0.1687, 0.2530, 25.003], label='Model 2')
#
# plt.xlabel('Training Sentences (in Millions)')
# plt.ylabel('Memory Consumed (in MBs)')
# plt.title("Training Size vs Memory")
# plt.axis([0, 9.5, 100, 1200])
# plt.legend()
# plt.show()

plt.plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9753], [75.8, 79.44, 81.58, 81.59, 83.5, 83.6, 83.87, 84.26, 84.74, 84.88], label='submit.jar')
plt.plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9753], [75.13, 80.17, 81.84, 81.84, 83.2, 83.62, 83.49, 84.07, 84.26, 85.14], label='best.jar')

plt.xlabel('Number of Training Sentences')
plt.ylabel('F1 score')
plt.title("Training Size v/s F1 Score on maxSentenceLength = 15")
plt.axis([1000, 10000, 74, 86])
plt.legend()
plt.show()
