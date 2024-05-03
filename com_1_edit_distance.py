def com_1_edit_distance(word1, word2):
    if len(word2) - len(word1) not in [-1, 0, 1]: return
    edit = 0
    cor1 = cor2 = action = None
    if len(word1) < len(word2):
        word1, word2 = word2, word1
        action = "insertion"

    if len(word1) == len(word2):
        mistakes = []
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                edit += 1
                cor1 = i
                mistakes.append(i)
                action = "substitution"
                if edit > 2:
                    return
        if edit == 2:
            if (
                word1[mistakes[0]] == word2[mistakes[1]] and
                word2[mistakes[0]] == word1[mistakes[1]]
            ):
                cor1 = mistakes[0]
                cor2 = mistakes[1]
                action = "transposition"
                edit = 1
    else:
        i = 0
        j = 0
        if action is None:
            action = "deletion"
        
        word2 += "^"
        
        word1 = list(word1)
        word2 = list(word2)

        while i < len(word1) and j < len(word2):
            if word1[i] != word2[j]:
                cor1 = i
                i += 1
                edit += 1
                if edit > 1:
                    return
            else:
                word1[i] = ">"
                word2[j] = "<"
                i += 1
                j += 1

    if edit == 1:
        return edit, action, cor1, cor2