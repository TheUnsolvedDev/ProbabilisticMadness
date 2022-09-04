import networkx as nx 
import matplotlib.pyplot as plt

if __name__ == '__main__':
	model = nx.DiGraph()
	model.add_edge("N","H")
	model.add_edge("W","H")
	model.add_edge("H","P")
	model.add_edge("W","O")

	nx.draw_networkx(model)
	plt.show()

	print(nx.d_separated(model,{"H"},{"O"},{"W"}))
	print(nx.d_separated(model,{"W"},{"P"},{"H"}))
	print(nx.d_separated(model,{"N"},{"W"},{"H"}))