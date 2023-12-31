function GloVeEmbedding:stemEmbeddingFile(embedding_file)
	local fstemtable={}
	local bstemtable={}
	local nostemtable={}
	for line in io.lines(embedding_file) do 
		local parts = split(line, " ") 
		local word = parts[1]
		local thestem = stem(word)
		fstemtable[word]=stem
		if type(bstemtable[stem]) == nil then
			bstemtable[stem]={}
		end	
		table.insert(bstemtable[stem],word)
		table.insert(nostemtable,word)
	end
	for stem, wordlist in pairs(bstemtable) do
		table.sort(wordlist)
		bstemtable[stem]=wordlist
	end
	table.sort(nostemtable)
	bstemtable["NotFound"]=nostemtable
	return bstemtable
end


function GloVeEmbedding:getNearestWord(word,bstemtable)
	local thestem=stem(word)
	local wordlist={}
	local nearestword
	if type(bstemtable[thestem]) ~= nil then
		wordlist=bstemtable[thestem]
	else
		wordlist=bstemtable["NotFound"]
	end
	for idx, listword in ipairs(wordlist)
		if (word>listword) then
			nearestword=word
			return nearestword
		end
	end
	nearestword="neutral";
	print("There has been a lookup error for this word: ", word)
	return nearestword
end
	
