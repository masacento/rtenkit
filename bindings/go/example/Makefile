.PHONY: static dynamic clean modeldata

MODEL_URL ?= https://github.com/masacento/rtenkit/releases/download/v0.1.2/all-MiniLM-L6-v2.zip
MODEL_ZIP = model.zip
MODEL_DIR = modeldata

static: example_static

dynamic: example_dynamic

example_static:
	go build -tags rtenkit_static -o example_static .

example_dynamic:
	go build -o example_dynamic .

modeldata: $(MODEL_DIR)

$(MODEL_DIR): $(MODEL_ZIP)
	unzip $(MODEL_ZIP) -d $(MODEL_DIR)

$(MODEL_ZIP):
	curl -L $(MODEL_URL) -o $(MODEL_ZIP)

clean:
	rm -f example_static example_dynamic
	rm -f $(MODEL_ZIP)
	rm -rf $(MODEL_DIR)
