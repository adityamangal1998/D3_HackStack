# cropped Face image window

bg_orig = Image.open('images/face.png')
bg_orig = bg_orig.resize((300, 300), Image.ANTIALIAS)
bg_orig = ImageTk.PhotoImage(bg_orig)
original_window = Label(main_frame, image=bg_orig)
original_window.place(x=0, y=0, height=300, width=300)

# contour image window

bg_contour = Image.open('images/face.png')
bg_contour = bg_contour.resize((300, 300), Image.ANTIALIAS)
bg_contour = ImageTk.PhotoImage(bg_contour)
contour_window = Label(main_frame, image=bg_contour)
contour_window.place(x=0, y=300, height=300, width=300)

# mesh cam window

bg_mesh = Image.open('images/face.png')
bg_mesh = bg_mesh.resize((300, 300), Image.ANTIALIAS)
bg_mesh = ImageTk.PhotoImage(bg_mesh)
mesh_window = Label(main_frame, image=bg_mesh)
mesh_window.place(x=0, y=600, height=300, width=300)

# canvas cam window

bg_canvas = Image.open('images/canvas.png')
bg_canvas = bg_canvas.resize((800, 300), Image.ANTIALIAS)
bg_canvas = ImageTk.PhotoImage(bg_canvas)
canvas_window = Label(main_frame, image=bg_canvas)
canvas_window.place(x=300, y=600, height=300, width=800)

# headBound cam window

bg_headBound = Image.open('images/face.png')
bg_headBound = bg_headBound.resize((300, 300), Image.ANTIALIAS)
bg_headBound = ImageTk.PhotoImage(bg_headBound)
headBound_window = Label(main_frame, image=bg_headBound)
headBound_window.place(x=1100, y=600, height=300, width=300)

# headposDetail cam window

bg_headPosDetail = Image.open('images/face.png')
bg_headPosDetail = bg_headPosDetail.resize((300, 300), Image.ANTIALIAS)
bg_headPosDetail = ImageTk.PhotoImage(bg_headPosDetail)
headPosDetail_window = Label(main_frame, image=bg_headPosDetail)
headPosDetail_window.place(x=1400, y=600, height=300, width=300)

# person cam window

bg_person = Image.open('images/person.png')
bg_person = bg_person.resize((800, 600), Image.ANTIALIAS)
bg_person = ImageTk.PhotoImage(bg_person)
person_window = Label(main_frame, image=bg_person)
person_window.place(x=300, y=0, height=600, width=800)


# eyeGraph cam window

bg_eyeGraph = Image.open('images/face.png')
bg_eyeGraph = bg_eyeGraph.resize((300, 300), Image.ANTIALIAS)
bg_eyeGraph = ImageTk.PhotoImage(bg_eyeGraph)
eyeGraph_window = Label(main_frame, image=bg_eyeGraph)
eyeGraph_window.place(x=1100, y=0, height=300, width=300)

# mouthGraph cam window

bg_mouthGraph = Image.open('images/face.png')
bg_mouthGraph = bg_mouthGraph.resize((300, 300), Image.ANTIALIAS)
bg_mouthGraph = ImageTk.PhotoImage(bg_mouthGraph)
mouthGraph_window = Label(main_frame, image=bg_mouthGraph)
mouthGraph_window.place(x=1100, y=300, height=300, width=300)


# mouthGraph Detail cam window

bg_mouthGraphDetail = Image.open('images/face.png')
bg_mouthGraphDetail = bg_mouthGraphDetail.resize((300, 300), Image.ANTIALIAS)
bg_mouthGraphDetail = ImageTk.PhotoImage(bg_mouthGraphDetail)
mouthGraphDetail_window = Label(main_frame, image=bg_mouthGraphDetail)
mouthGraphDetail_window.place(x=1400, y=300, height=300, width=300)

# eyeGraph Detail window

bg_eyeGraphDetail = Image.open('images/face.png')
bg_eyeGraphDetail = bg_eyeGraphDetail.resize((300, 300), Image.ANTIALIAS)
bg_eyeGraphDetail = ImageTk.PhotoImage(bg_eyeGraphDetail)
eyeGraphDetail_window = Label(main_frame, image=bg_eyeGraphDetail)
eyeGraphDetail_window.place(x=1400, y=0, height=300, width=300)
