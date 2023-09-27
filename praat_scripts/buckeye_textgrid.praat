###################
#
# Create TextGrids for all the sound files in the Buckeye Corpus.
# Created by bvt Apr. 2011
#
#############


form Give the parameters for pause analysis
   comment This script marks the pauses in the Sound file and creates an IntervalTier indicating the silence and the word in a TextGrid. It then extracts the word without silences on either end.
   comment The path of the directory - choose from list:
    optionmenu Directory: 1
        option Choose from list
        option C:\Users\mckelley\buckeye\buckeye\
   comment or give the path - e.g. 'd:\sound\':
    text Directory_manual
   comment Specify the slash direction "\" for Windows and "/" for everything else.
    text Slash

endform

if length(directory_manual$) > 0
    directory$ = directory_manual$
endif


Create Strings as directory list... directoryList 'directory$'*
	Change... . "" 0 Literals
	Rename...  directoryList1
	select Strings directoryList
	Remove
	select Strings directoryList1
	Rename...  directoryList2
	select Strings directoryList2

numberOfdirs = Get number of strings

for dir to numberOfdirs
dirname$ = Get string... dir
	if dirname$ <> ""
	Create Strings as file list... list 'directory$''dirname$''slash$'*.wav
		Change... .wav "" 0 Literals
		Rename...  list5

		select Strings list
		Remove
		select Strings list5

		numberOfFiles = Get number of strings
		for ifile to numberOfFiles
		    soundname$ = Get string... ifile
	  		if soundname$ <> ""
				Read IntervalTier from Xwaves... 'directory$''dirname$''slash$''soundname$'.words
				Read IntervalTier from Xwaves... 'directory$''dirname$''slash$''soundname$'.phones
				Rename... 'soundname$'_phones
				select IntervalTier 'soundname$'_phones
				plus IntervalTier 'soundname$'
				Into TextGrid
				Write to text file... 'directory$''dirname$''slash$''soundname$'.TextGrid
				
				select IntervalTier 'soundname$'_phones
				plus IntervalTier 'soundname$'
				plus TextGrid grid
				Remove
				select Strings list5
			endif
		endfor

		select Strings list5
		Remove
	endif

	select Strings directoryList2
endfor

select Strings directoryList2
Remove

echo C'est finis!

