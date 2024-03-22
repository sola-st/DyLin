import subprocess

def install_special(url):
    if url == "https://github.com/lorien/grab.git":
        command = "pip install cssselect pyquery pymongo fastrq"  # required for running tests
    elif url == "https://github.com/psf/black.git":
        command = "pip install aiohttp"  # required for running tests
    elif url == "https://github.com/errbotio/errbot.git":
        command = "pip install mock"  # required for running tests
    elif url == "https://github.com/PyFilesystem/pyfilesystem2.git":
        command = "pip install parameterized pyftpdlib psutil"  # required for running tests
    elif url == "https://github.com/wtforms/wtforms.git":
        command = "pip install babel email_validator"  # required for running tests
    elif url == "https://github.com/geopy/geopy.git":
        command = "pip install docutils"  # required for running tests
    elif url == "https://github.com/gawel/pyquery.git":
        command = "pip install webtest"  # required for running tests
    elif url == "https://github.com/elastic/elasticsearch-dsl-py.git":
        command = "pip install pytz"  # required for running tests
    elif url == "https://github.com/marshmallow-code/marshmallow.git":
        command = "pip install pytz simplejson"  # required for running tests
    elif url == "https://github.com/pytest-dev/pytest.git":
        command = "pip install hypothesis xmlschema"  # required for running tests
    elif url == "https://github.com/miso-belica/sumy.git":
        subprocess.run(["pip", "install", "nltk"])
        command = "python -m nltk.downloader all"
    elif url == "https://github.com/python-telegram-bot/python-telegram-bot.git":
        command = "pre-commit install"
    elif url == "https://github.com/dpkp/kafka-python.git":
        subprocess.run(["apt-get", "install", "-y", "libsnappy-dev"])
        command = "pip install pytest-mock mock python-snappy zstandard lz4 xxhash crc32c"
    elif url == "https://github.com/sphinx-doc/sphinx.git":
        command = "pip install html5lib"
    elif url == "https://github.com/Trusted-AI/adversarial-robustness-toolbox.git":
        command = "pip install Pillow"
    elif url == "https://github.com/spotify/dh-virtualenv.git":
        command = "pip install mock nose tf_keras"
    elif url == "https://github.com/Suor/funcy.git":
        command = "pip install more-itertools whatever"
    elif url == "https://github.com/WebOfTrust/keripy.git":
        command = "pip install lmdb pysodium blake3 msgpack simplejson cbor2"
    else:
        return
    subprocess.run(command.split(" "))

