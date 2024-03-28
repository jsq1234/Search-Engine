import ir_datasets

dt = ir_datasets.load("beir/trec-covid")

for (id, item) in enumerate(dt.docs_iter()):
    print(item)
    if id > 5:
        break
