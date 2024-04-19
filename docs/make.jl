using POD_LES
using Documenter

DocMeta.setdocmeta!(POD_LES, :DocTestSetup, :(using POD_LES); recursive=true)

makedocs(;
    modules=[POD_LES],
    authors="tobyvg <tobyvangastelen@gmail.com> and contributors",
    sitename="POD_LES.jl",
    format=Documenter.HTML(;
        canonical="https://tobyvg.github.io/POD_LES.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tobyvg/POD_LES.jl",
    devbranch="main",
)
