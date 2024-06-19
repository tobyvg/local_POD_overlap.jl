using local_POD_overlap
using Documenter

DocMeta.setdocmeta!(local_POD_overlap, :DocTestSetup, :(using local_POD_overlap); recursive=true)

makedocs(;
    modules=[local_POD_overlap],
    authors="tobyvg <tobyvangastelen@gmail.com> and contributors",
    sitename="local_POD_overlap.jl",
    format=Documenter.HTML(;
        canonical="https://tobyvg.github.io/local_POD_overlap.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tobyvg/local_POD_overlap.jl",
    devbranch="main",
)
