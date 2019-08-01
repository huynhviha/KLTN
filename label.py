def read_data(path):
	f = open(path, "r+")
	new = open(path + "1", "w+")
	for img in f:
		labels = open("Densenet121/label.txt", "r+")
		for label in labels:
			l = label.split(" ")
			a, b = l[0].split("/")
			if b in img:
				new.write(label)
				break
		labels.close()
read_data("Densenet121/val_list.txt")