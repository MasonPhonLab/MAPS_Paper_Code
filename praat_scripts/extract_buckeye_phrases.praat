form Extract Buckeye phrases
	comment Buckeye directory (containing all speaker folders)
		text buckDir C:\Users\mckelley\buckeye\buckeye
	comment Output directory:
		text outDir C:\Users\mckelley\alignerv2\buck_out
endform

Create Strings as directory list: "dirList", buckDir$
select Strings dirList
nDirs = Get number of strings

for dirI from 1 to nDirs
	currDir$ = Get string: dirI
	@joinpath: buckDir$, currDir$
	currDir$ = joinpath.path$
	@joinpath: currDir$, "*.wav"
	globPath$ = joinpath.path$
	Create Strings as file list: "filenames", globPath$
	select Strings filenames
	nFiles = Get number of strings

	for fileI from 1 to nFiles
		fileName$ = Get string: fileI
		@joinpath: currDir$, fileName$
		filePath$ = joinpath.path$
		@extractSounds: filePath$

		# reset loop state
		select Strings filenames
	endfor


	# clean up filenames object
	select Strings filenames
	Remove

	# reset loop state
	select Strings dirList
endfor

# clean up directory list
select Strings dirList
Remove

procedure joinpath: a$, b$
	if not endsWith(a$, "/") and not endsWith(a$, "\")
		a$ = a$ + "/"
	endif
	if endsWith(b$, "/") or endsWith(b$, "\")
		bLen = length(b$)
		b$ = left$(b$, bLen - 1)
	endif
	.path$ = a$ + b$
endproc

procedure extractSounds: soundname$
	Read from file: soundname$
	tgname$ = replace$(soundname$, ".wav", ".TextGrid", 0)

	Read from file: tgname$

	objectName$ = selected$("TextGrid")

	select TextGrid 'objectName$'
	nIntervals = Get number of intervals: 1

	foundStart = 0
	transcriptEnded = 0
	for intervalI from 1 to nIntervals
		writeInfoLine: intervalI
		currLabel$ = Get label of interval: 1, intervalI
		currStart = Get start time of interval: 1, intervalI
		currEnd = Get end time of interval: 1, intervalI
		currDur = currEnd - currStart

		if index(currLabel$, "E_TRANS")
			transcriptEnded = 1
		endif
		
		if currDur > 0.010 and not foundStart and not transcriptEnded and not (index(currLabel$, "<") or index(currLabel$, "{") or index(currLabel$, "noise"))
			foundStart = 1
			soundStart = Get start time of interval: 1, intervalI

			# Get phone boundary because the word boundaries don't always match
			pIntervalN = Get interval at time: 2, soundStart + 0.010
			soundStart = Get start time of interval: 2, pIntervalN

		elif foundStart and not transcriptEnded and (index(currLabel$, "<") or index(currLabel$, "{") or index(currLabel$, "noise"))

			soundLast = Get end time of interval: 1, intervalI - 1
			pIntervalN = Get interval at time: 2, soundLast - 0.010
			soundLast = Get end time of interval: 2, pIntervalN

			# write TextGrid to file
			Extract part: soundStart, soundLast, "no"
			@joinpath: outDir$, objectName$ + "_'intervalI'" + ".TextGrid"
			outTgname$ = joinpath.path$
			Save as text file: outTgname$

			# write Sound to file
			select Sound 'objectName$'
			Extract part: soundStart, soundLast, "rectangular", 1.0, "no"
			@joinpath: outDir$, objectName$ + "_'intervalI'" + ".wav"
			outSoundname$ = joinpath.path$
			Save as WAV file: outSoundname$
			

			# Clean up TextGrid and Sound part objects
			select TextGrid 'objectName$'_part
			Remove
			select Sound 'objectName$'_part
			Remove

			# restore state to loop
			foundStart = 0
			select TextGrid 'objectName$'
		endif
	endfor

	select TextGrid 'objectName$'
	Remove
	select Sound 'objectName$'
	Remove
endproc

writeInfoLine: "DONE! :)"