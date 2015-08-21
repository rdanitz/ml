using DataFrames

include("porter.jl")

function processEmail(email)
  df = readtable("vocab.txt", separator='\t', header=false)
  vocab = convert(Vector, df[:x2])

  email = lowercase(email)
  email = replace(email, r"<[^<>]+>", " ")
  email = replace(email, r"[0-9]+", "number")
  email = replace(email, r"(http|https)://[^\s]*", "httpaddr")
  email = replace(email, r"[^\s]+@[^\s]+", "emailaddr")
  email = replace(email, r"[$]+", "dollar")

  sep = r"[ @$/#.-:&*+=\[\]?!(){},'\">_<;%\10\13\n]"
  tokens = filter(i->!isempty(i), split(email, sep))
  tokens = map(i->filter(isalnum, i), tokens)
  tokens = map(stem, tokens)
  tokens = filter(i->!isempty(i), tokens)

  filter(i->i!=0, indexin(tokens, vocab))
end
