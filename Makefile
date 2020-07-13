.PHONY: tag-release

tag-release:
	git tag "$(shell python setup.py --version)"
