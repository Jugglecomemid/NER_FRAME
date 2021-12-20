import setuptools

setuptools.setup(
    name="ner",
    version="0.0.1",
    author="qliu",
    author_email="liuqiantx1221@163.com",
    description="three NER models: BERT & BERT CRF & BERT BILSTM CRF",
    url="https://gitlab.mvalley.com/ir/websites",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)