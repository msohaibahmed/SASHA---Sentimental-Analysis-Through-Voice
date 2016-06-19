import moviepy.editor as mp

sub=1
sen=1
des="extract"
a=46


while sub!=45:
    src=des+"/s"+str(sub)+"_an_"+str(sen)+".avi"
    dst="wavFiles/a"+str(a)+".wav"
    print(src)
    clip = mp.VideoFileClip(str(src))
    clip.audio.write_audiofile(str(dst))
    if sen==5:
	sub+=1
        sen=0
    sen+=1
    if sub==6:
        sub+=1
    a+=1
    

print("done")
