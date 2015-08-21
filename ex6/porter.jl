###
### Porter stemmer by David Ellison
### https://github.com/dellison/CompLing.jl
###

### Implementation of the Porter stemming algorithm

type Word
    b::String
    k::Int
    k0::Int
    j::Int
end

Word(s::String) = Word(s, length(s), 1, 0)

irregular_words = {"sky" => "sky",
                   "skies" => "sky",
                   "dying" => "die",
                   "lying" => "lie",
                   "tying" => "tie",
                   "news" => "news",
                   "innings" => "inning",
                   "inning" => "inning",
                   "outing" => "outings",
                   "outing" => "outing",
                   "canning" => "cannings",
                   "canning" => "canning",
                   "howe" => "howe",
                   # --NEW--
                   "proceed" => "proceed",
                   "exceed" => "exceed",
                   "succeed" => "succeed" # Hiranmay Ghosh
                   }
                   
function is_cons(wd::Word, i::Int)
    ## is_cons(wd, i) is true <=> wd.b[i] is a consonant.
    for c in "aeiou"
        if wd.b[i] == c
            return false
        end
    end
    if wd.b[i] == 'y'
        if i == wd.k0
            return true
        else
            return ! is_cons(wd, i-1)
        end
    else
        return true
    end
end

function m(wd::Word)
    ## m() measures the number of consonant sequences between k0 and j.
    ## if c is a consonant sequence and v a vowel sequence, and <..>
    ## indicates arbitrary presence,
    ##    <c><v>       gives 0
    ##    <c>vc<v>     gives 1
    ##    <c>vcvc<v>   gives 2
    ##    <c>vcvcvc<v> gives 3
    ##    ....
    local n = 0
    ## local i = wd.k0
    local i = 1 # nltk
    while true
        if i > wd.j
            return n
        end
        if !is_cons(wd, i)
            break
        end
        i += 1
    end
    i += 1
    while true
        while true
            if i > wd.j
                return n
            end
            if is_cons(wd, i)
                break
            end
            i += 1
        end
        i += 1
        n += 1
        while true
            if i > wd.j
                return n
            end
            if ! is_cons(wd, i)
                break
            end
            i += 1
        end
        i += 1
    end
    return i
end
    
function vowelinstem(wd::Word)
    for i in wd.k0:wd.j # + 1 (?)
        if !is_cons(wd, i)
            return true
        end
    end
    return false
end

function doublecons(wd::Word, j)
    if j < wd.k0 + 1
        return false
    end
    if wd.b[j] != wd.b[j-1]
        return false
    end
    return is_cons(wd, j)
end

function cvc(wd::Word, i::Int)
    ## cvc(i) is TRUE <=>
    ## a) ( --NEW--) i == 1, and p[0] p[1] is vowel consonant, or
    ## b) p[i - 2], p[i - 1], p[i] has the form consonant -
    ##    vowel - consonant and also if the second c is not w, x or y. this
    ##    is used when trying to restore an e at the end of a short word.
    ##    e.g.
    ##        cav(e), lov(e), hop(e), crim(e), but
    ##        snow, box, tray.
    if i == 0
        return false # never happens?
    end
    if i == 1
        ## FIXME?
        ## return ! is_cons(wd, i) || is_cons(wd, i-1) || ! is_cons(wd, i-2)
        return ( !cons(wd, 1) && cons(wd, 2) )
    end
    if (i < wd.k0+2) || !is_cons(wd, i) || is_cons(wd, i-1) || !is_cons(wd, i-2)
        ## print("*")
        if i < wd.k0+2
            ## print("*")
            return true #?????
        end
        return false
    end
    ch = wd.b[i]
    if ch == 'w' || ch == 'x' || ch == 'y'
        return false
    end
    return true

    ## 
    ## if i < (wd.k0 + 2) || ! cons(wd, i) || cons(wd, i-1) || !cons(wd, i-2)
    ##     return false
    ## end
    ## ch = wd.b[i]
    ## if ch =='w' || ch =='x' || ch =='y'
    ##     return false
    ## end
    ## return true
end

function ends(wd::Word, s::String)
    # ends(s) returns TRUE <=> k0,...k ends with the string s.
    local len = length(s)
    if s[len] != wd.b[wd.k]
        return false
    end
    if len > (wd.k - wd.k0 + 1)
        return false
    end
    if wd.b[wd.k-len+1:wd.k] != s
        return false
    end
    wd.j = wd.k - len
    return true    
end

function setto(wd::Word, s::String)
    # setto(s) sets (j+1),...k to the characters in the string s, readjusting k.
    local len = length(s)
    wd.b = string(wd.b[1:wd.j], s, wd.b[wd.j+len+1:end]) # added +1
    wd.k = wd.j + len
    return wd
end

function r(wd::Word, s::String)
    if m(wd) > 0
        wd = setto(wd, s)
    end
    return wd
end

function step1ab(wd::Word)
    ## step1ab() gets rid of plurals and -ed or -ing. e.g.
    ## caresses  ->  caress
    ## ponies    ->  poni
    ## sties     ->  sti
    ## tie       ->  tie        (--NEW--: see below)
    ## caress    ->  caress
    ## cats      ->  cat
    ## feed      ->  feed
    ## agreed    ->  agree
    ## disabled  ->  disable
    ## matting   ->  mat
    ## mating    ->  mate
    ## meeting   ->  meet
    ## milling   ->  mill
    ## messing   ->  mess
    ## meetings  ->  meet
    if wd.b[wd.k] == 's'
        if ends(wd, "sses")
            wd.k -= 2
        elseif ends(wd, "ies")
            if wd.j == 1 # 0
                wd.k -= 1
            else
                wd.k -= 2
            end
            ## wd = setto(wd, "i")
        elseif wd.b[wd.k - 1] != 's'
            wd.k = wd.k - 1
        end
    end
    if ends(wd, "ied")
        if wd.j == 1
            wd.k -= 1
        else
            wd.k -= 2
        end
    elseif ends(wd, "eed")
        if m(wd) > 0
            wd.k -= 1
        end
    elseif (ends(wd, "ed") || ends(wd, "ing")) && vowelinstem(wd) # grouping?
        wd.k = wd.j
        if ends(wd, "at")
            setto(wd, "ate")
        elseif ends(wd, "bl")
            setto(wd, "ble")
        elseif ends(wd, "iz")
            setto(wd, "ize")
        elseif doublecons(wd, wd.k) # was j
            wd.k -= 1
            ch = wd.b[wd.k]
            if ch == 'l' || ch == 's' || ch == 'z'
                wd.k += 1
            end
        elseif m(wd) == 1 && cvc(wd, wd.k)
            ## println("here")
            wd = setto(wd, "e")
        end
    end
    return wd
end

function step1c(wd::Word)
    ## porter_1c() turns terminal y to i when there is another vowel in the stem.
    ## --NEW--: This has been modified from the original Porter algorithm so that y->i
    ## is only done when y is preceded by a consonant, but not if the stem
    ## is only a single consonant, i.e.
    ##    (*c and not c) Y -> I
    ## So 'happy' -> 'happi', but
    ##   'enjoy' -> 'enjoy'  etc
    ## This is a much better rule. Formerly 'enjoy'->'enjoi' and 'enjoyment'->
    ## 'enjoy'. Step 1c is perhaps done too soon; but with this modification that
    ## no longer really matters.
    ## Also, the removal of the vowelinstem(z) condition means that 'spy', 'fly',
    ## 'try' ... stem to 'spi', 'fli', 'tri' and conflate with 'spied', 'tried',
    ## 'flies' ...
    if ends(wd, "y") && wd.j > 1 && is_cons(wd, wd.k-1)
    ## if ends(wd, "y") && vowelinstem(wd) # this was the original porter algorithm
        wd.b = string(wd.b[1:wd.k-1], 'i', wd.b[wd.k:end])
    end
    return wd
end

function step2(wd::Word)
    ## porter_2() maps double suffices to single ones.
    ## so -ization ( = -ize plus -ation) maps to -ize etc. note that the
    ## string before the suffix must give m() > 0.
    if wd.k == 1
        wd.k += 1
    end
    if wd.b[wd.k - 1] == 'a'
        if ends(wd, "ational")
            wd = r(wd, "ate")
        elseif ends(wd, "tional")
            wd = r(wd, "tion")
        end
    elseif wd.b[wd.k - 1] == 'c'
        if ends(wd, "enci")
            wd = r(wd, "ence")
        elseif ends(wd, "anci")
            wd = r(wd, "ance")
        end
    elseif wd.b[wd.k-1] == 'e'
        if ends(wd, "izer")
            wd = r(wd, "ize")
        end
    elseif wd.b[wd.k - 1] == 'l'
        if ends(wd, "bli")
            wd = r(wd, "ble")
        elseif ends(wd, "alli")
            if m(wd) > 0
                wd = setto(wd, "al") ## NEW in nltk
                wd = step2(wd)
            end
        elseif ends(wd, "fulli") ## NEW in nltk
            wd = r(wd, "ful")
        elseif ends(wd, "entli")
            wd = r(wd, "ent")
        elseif ends(wd, "eli")
            wd = r(wd, "e")
        elseif ends(wd, "ousli")
            wd = r(wd, "ous")
        end
    elseif wd.b[wd.k - 1] == 'o'
        if ends(wd, "ization")
            wd = r(wd, "ize")
        elseif ends(wd, "ation")
            wd = r(wd, "ate")
        elseif ends(wd, "ator")
            wd = r(wd, "ate")
        end
    elseif wd.b[wd.k - 1] == 's'
        if ends(wd, "alism")
            wd = r(wd, "al")
        elseif ends(wd, "iveness")
            wd = r(wd, "ive")
        elseif ends(wd, "fulness")
            wd = r(wd, "ful")
        elseif ends(wd, "ousness")
            wd = r(wd, "ous")
        end
    elseif wd.b[wd.k - 1] == 't'
        if ends(wd, "aliti")
            wd = r(wd, "al")
        elseif ends(wd, "iviti")
            wd = r(wd, "ive")
        elseif ends(wd, "biliti")
            wd = r(wd, "ble")
        end
    elseif wd.b[wd.k - 1] == 'g' # departure
        if ends(wd, "logi")
            wd.j += 1
            r(wd, "og")
        end
    end
    return wd
end

function step3(wd::Word)
    # porter_3() deals with -ic-, -full, -ness, etc.
    # similar strategy to porter_2.
    if wd.b[wd.k] == 'e'
        if ends(wd, "icate")
            wd = r(wd, "ic")
        elseif ends(wd, "ative")
            wd = r(wd, "")
        elseif ends(wd, "alize")
            wd = r(wd, "al")
        end
    elseif wd.b[wd.k] == 'i'
        if ends(wd, "iciti")
            wd = r(wd, "ic")
        end
    elseif wd.b[wd.k] == 'l'
        if ends(wd, "ical")
            wd = r(wd, "ic")
        elseif ends(wd, "ful")
            wd = r(wd, "")
        end
    elseif wd.b[wd.k] == 's'
        if ends(wd, "ness")
            wd = r(wd, "")
        end
    end
    return wd
end

function step4(wd::Word)
    # porter_4() takes off -ant, -ence, etc.,
    # in context <c>vcvc<v>
    if wd.b[wd.k - 1] == 'a'
        #####
        ## TODO: clean these up... if X: pass is ugly
        #####
        if  ends(wd, "al")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'c'
        if ends(wd, "ance")
            # do nothing
        elseif ends(wd, "ence") # this is in nltk
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'e'
        if ends(wd, "er")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'i'
        if ends(wd, "ic")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'l'
        if ends(wd, "able")
            # do nothing
        elseif ends(wd, "ible")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'n'
        if ends(wd, "ant")
            # do nothing
        elseif ends(wd, "ement")
            # do nothing
        elseif ends(wd, "ment")
            # do nothing
        elseif ends(wd, "ent")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'o'
        if ends(wd, "ion") && (wd.b[wd.j] == 's' || wd.b[wd.j] == 't')
            # do nothing
        elseif ends(wd, "ou")
            # do nothing
            # takes care of -ous
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 's'
        if ends(wd, "ism")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 't'
        if ends(wd, "ate")
            # do nothing
        elseif ends(wd, "iti")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'u'
        if ends(wd, "ous")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'v'
        if ends(wd, "ive")
            # do nothing
        else
            return wd
        end
    elseif wd.b[wd.k - 1] == 'z'
        if ends(wd, "ize")
            # do nothing
        else
            return wd
        end
    else
        return wd
    end
    if m(wd) > 1
        wd.k = wd.j
    end
    return wd
end

function step5(wd::Word)
    #step5(word) removes a final -e if m(word) > 1, and changes -ll to -l if m(word > 1)
    wd.j = wd.k
    if wd.b[wd.k] == 'e'
        a = m(wd)
        if a > 1 || (a == 1 && !cvc(wd, wd.k-1))
            wd.k -= 1
        end
    end
    if wd.b[wd.k] == 'l' && doublecons(wd, wd.k) && m(wd) > 1
        wd.k -= 1
    end
    return wd
end

function stem(wd::Word)
    wd = step1ab(wd)
    wd = step1c(wd)
    wd = step2(wd)
    wd = step3(wd)
    wd = step4(wd)
    wd = step5(wd)
    return wd.b[wd.k0:wd.k]
end

function stem(s::String)
    if in(s, keys(irregular_words))
        return irregular_words[s]
    elseif length(s) >= 3
        return stem(Word(s, length(s), 1, 0))
    else
        return s
    end
end
