## New in this build

Your precompiled shaders stay valid: for as long as you keep them, the shaders are translated as in v20261023. Your
next precompile (a new install, or after a graphics driver update) brings lighter shaders (first item).

- **Lighter shaders** (with your next precompile): a PS5 shader's register writes are masked lane by lane, and the
  emulator kept the old value of every lane outside the mask with an extra instruction: 20-32% of all instructions in
  the heaviest shaders. Where nothing reads those lanes, that instruction is now left out. The game's translated
  shaders are 8.5% smaller, the heaviest compute shaders run 4-14% fewer instructions on NVIDIA (one of them 128
  registers instead of 168), and an RTX 5090 at 4K does 3% less GPU work a frame at Boletaria: up to 3% more fps where
  the graphics card is the limit. Shaders precompiled by earlier versions stay in use and up to date, so no precompile
  is offered for this alone. `run.cmd -Set KYTY_EXEC_SELECTS=0` turns it off (precompile again after that).
- **VRR displays** (#221): on a G-SYNC or FreeSync display, put `-Set KYTY_VRR=1` in the launcher's Advanced page,
  Command-line options (or `run.cmd -Set KYTY_VRR=1`). Each frame is then shown as soon as it is done, as a PS5 does on
  a VRR display, instead of at the next tick of the emulator's 60 Hz clock. Below 60 fps the frames stayed on screen
  for 16.7 or 33.3 ms by turns while the game moved by its real frame time: uneven motion even on a VRR display. At
  1-1 with the graphics card as the limit, the difference between one frame's time on screen and the next went from
  8.3 ms to 0.9 ms. Off by default; the log's `Flips:` line every 30 s shows how even the frames were.
- **Freezes after menus at 120 fps** (#221): with the 120 fps option and the card's video memory full, a menu kept open
  for 5 seconds let the textures of the world be dropped, and they were made again as it closed: frames of 1-5 s
  (about 1.3 GB). The texture collector now counts its ages in seconds (10 and 20 s) at any frame rate.
- **No anti-aliasing** (#220): the launcher's Upscaler list has "Off (no anti-aliasing)": Demon's Souls without its
  TAA and the camera jitter TAA needs, a sharper picture with jagged edges (`run.cmd -Antialiasing off`).
- **Video memory for frame generation and upscaling** (#215): DLSS frame generation, Linux's built-in FSR 3 frame
  generation, and DLSS or FSR 3 upscaling now get room in the card's video memory budget (1.5 GB for frame
  generation, 0.5 GB more for upscaling), as FSR frame generation on Windows already did. An RTX 5080 (16 GB) with DLSS
  and DLSS frame generation at 3440x1440 had counted as a card with plenty of memory and ran out of it late in a
  session. `KYTY_FSR_VRAM_RESERVE_MB` sets the frame generation share; the log says what was set aside.
- **Less RAM with Texture reduce**: Texture reduce (on by default for cards under 12 GB) keeps the largest level of
  big textures out of video memory; now the game no longer loads those levels either. At Boletaria the textures it
  streams went from 2.7 to 1.8 GB, about 1 GB less RAM, with the same picture.
- **Memory back to Windows sooner on 16 GB PCs** (#174): after a trip between worlds, the memory the game freed goes
  back to Windows twice as fast (2.3 GB instead of 1.1 GB on our test route). On a PC that pages hard, giving it back
  no longer holds the game up for up to a second at a time while Windows is slow.
- **Read-back pages on 16 GB PCs** (#159, #174): below 2 GB of free RAM, v20261026 protected the pages read back from
  the graphics card with guard pages (the blue screen fix). Guard pages let another thread read an old page now and
  then (a texture flickering for a moment), and 16 GB PCs sit around 2 GB free most of a session. They now take over
  only below 1.5 GB (every reported blue screen came with less than that) and give way again at 2 GB (was 3 GB).
- **Fewer readbacks on PCs with more than 20 GB of RAM** (PR #219, @DevList1454): when the game reads memory the
  graphics card wrote, neighbouring buffers now come back in the same copy. At Boletaria 1-3 the readbacks went from
  about 51 to 19 a frame, and the frame rate from 50.2 to 52.2 fps in @DevList1454's measurements.
- **Back to Big Picture** (#217): started from Steam's Big Picture, quitting the game hands the focus back to Big
  Picture, which takes the controller again without a click.
- **Steam Deck** (Linux package): the Deck's GPU now runs the game's 64-wide compute shaders natively (23% less GPU
  time in them; 20.6 to 22.1 fps at 1-1 with the GPU held at 1600 MHz), and the emulator's own GPU work for its memory
  tracking is lighter on every graphics card (7.2 to 0.9 ms a frame on the Deck). Not 30 fps yet.
- **Thanks** to @DevList1454 for PR #219, and to @crni20zd, @Psychowarden, @javsp, @ft1704, @thxnk420, @Myoko and @brokeneggg for
  their reports and tests.

## New in v20261026

Your precompiled shaders stay valid: the shaders are the same as in v20261023. The precompile only adds the pipelines of
the new shader hints (#212).

- **Crash on version 1.005** (#185, #204): on some PCs Demon's Souls 1.005 (European and US editions) crashed within
  minutes of play, or already at start-up. The game's resource loader could take a load request that another thread
  was withdrawing at that moment. The emulator now makes the loader wait until the withdrawal is finished; the log line
  `Demon's Souls loader guard: ... waited out` shows each time it did.
- **Blue screens on 16 GB PCs** (#91): since v20261024, the pages the game reads back from the graphics card were
  protected in a way that could crash Windows (MEMORY_MANAGEMENT) once the PC ran low on RAM. Whenever less than 2 GB
  of RAM is free they now get guard pages instead, which held up under heavy memory pressure in our tests.
- **Shader preparation kept on disk** (#200): the background shader preparation and the start-up warmup now keep their
  translations in `_PipelineCache\static\<title>_<version>.prefetch` (1-2 GB). The first start after this update
  translates as before; later starts read them in about a second. On our desktop the start-up warmup went from 22 s to
  1 s, and PCs with few cores no longer translate for minutes beside the game. `run.cmd -Set KYTY_PREFETCH_CACHE=0`
  turns it off. After a patch by @Psychowarden.
- **Lit ground and water flickering white** (#196): buffer copies made on a background thread could take data the game
  had already replaced for its next frame. The copies now finish before the game can see that the graphics card is
  done with them. Tracked down with @Psychowarden's tests.
- **Frame generation at camera cuts** (#174): when a dialogue or a cutscene shot ended, the frame generated in between
  blended the two camera views, both half-transparent. With an upscaler chosen (DLSS or FSR), frame generation now
  starts over at the game's camera cuts.
- **The 1440p output in the 4K layout** (#187, #202): at the 1440p output the interface is laid out as at 4K, at the
  output's size, instead of the 1080p layout scaled up: sharp text, and the character in the character creator and the
  equipment menu in place. `run.cmd -Set KYTY_OUTPUT_LAYOUT=hd` keeps the old way.
- **Wide screens at 1440p** (#199, #198): a 21:9 or wider aspect with the 1440p output now makes the picture at that
  height (3416x1440 at 21:9) instead of 16:9 (77 -> 90 fps on an RTX 5080 at 43:18), and movies are no longer squeezed
  with a wide output. After @Psychowarden's patches.
- **Audio output** (#207): the launcher's Game page has "Audio output": TV, headphones, AV amplifier or soundbar. The
  game is told your choice and mixes its sound for it, as on a PS5 (`run.cmd -Set KYTY_AUDIO_OUTPUT=headphones`). Found
  by @Psychowarden.
- **Quitting with a controller** (#203): hold the PS button (an Xbox pad's Guide button) for two seconds to quit, as
  Close Game does on a PS5. Where Steam or the Game Bar takes that button, `run.cmd -Set KYTY_PAD_QUIT=both` also quits
  on Create and Options (an Xbox pad's View and Menu) held together. An Xbox pad's View button now presses the touchpad
  (#197).
- **More shader hints** (#212): a playthrough to both endings added 237 shaders and 268 pipelines the release's hints
  lacked: fewer first-time stutters late in the game. Thanks to @Vorshim92.
- **Precompile** (#213): a shard that stops working no longer holds up the whole precompile. After 20 minutes without
  work it is started again, twice at most, and then left out; the game compiles its pipelines when it first needs them.
- **Less memory on 16 GB PCs**: memory the game's engine has freed goes back to Windows four times faster: 0.8 GB less
  in a long session.
- **New images start cleared** (PR #193, @intellq): as on the PS5, a new image is zero before its first use. With FSR
  at 1080p and the 43:18 squeeze, its first frame read undefined contents and kept a green or red picture for the whole
  session, mostly on Linux with AMD cards.
- **Linux**: AMD graphics cards (RADV) now get their transfer queue, so readbacks of finished GPU work no longer wait
  behind the graphics queue: +19% to +27% while walking on an RX 9060 XT and RX 9070 XT. Found by @the2ndday, tested by
  @dreamerk24m and @yazuc. `-Set KYTY_RADV_TRANSFER_QUEUE=0` turns it off; owners of older AMD cards, please report any
  GPU hang. Also from @the2ndday: an experimental `KYTY_UFFD_WP=1` (the GPU trackers' write watch by userfaultfd,
  Linux 6.4+; PR #206), and the bench helpers on Linux (PR #211).
- **Thanks** to @Psychowarden, @the2ndday, @Vorshim92 and @intellq for their patches and pull requests, and to
  @danielmazurek79, @Theoryedz, @ededaelita1-afk, @ft1704, @dreamerk24m, @Fantomas3228, @yazuc, @marcelofis,
  @SanyaSan and @kvothe3323 for their reports and tests.

## 本版本的变化

已有的预编译着色器继续有效：只要保留它们，着色器就按 v20261023 的方式翻译。下次预编译（新安装，或更新显卡驱动之后）会得到
更轻量的着色器（第一项）。

- **更轻量的着色器**（下次预编译后生效）：PS5 着色器写寄存器时按通道屏蔽，模拟器以前用一条额外指令为屏蔽之外的每个通道保留
  旧值，占最重着色器全部指令的 20-32%。现在只要没有地方读取这些通道，就省掉这条指令。游戏翻译后的着色器小了 8.5%，最重的
  计算着色器在 NVIDIA 上少执行 4-14% 的指令（其中一个寄存器从 168 降到 128），RTX 5090 在 4K、波雷塔利亚每帧的 GPU 工作少
  3%：在显卡是瓶颈的场合，帧率最多高约 3%。旧版本预编译的着色器继续使用、仍算最新，不会因为这一项提示重新预编译。
  `run.cmd -Set KYTY_EXEC_SELECTS=0` 可以关闭（关闭后请重新预编译）。
- **VRR 显示器**（#221）：在 G-SYNC 或 FreeSync 显示器上，把 `-Set KYTY_VRR=1` 填进启动器「高级」页的「命令行参数」（或
  `run.cmd -Set KYTY_VRR=1`）。每一帧画完就立即显示，和 PS5 在 VRR 显示器上一样，而不是等模拟器 60 Hz 时钟的下一拍。低于
  60 fps 时，以前每帧轮流停留 16.7 或 33.3 ms，而游戏按实际帧时间推进画面：即使在 VRR 显示器上，动作也不均匀。在 1-1、显卡
  为瓶颈时，相邻两帧显示时长之差从 8.3 ms 降到 0.9 ms。默认关闭；日志每 30 秒的 `Flips:` 行会显示帧是否均匀。
- **120 fps 下关闭菜单后的卡顿**（#221）：开启 120 fps 选项且显存已满时，菜单打开 5 秒就会让场景的纹理被回收，关闭菜单时
  再重新生成：出现 1-5 秒的帧（约 1.3 GB）。纹理回收现在在任何帧率下都按秒计算（10 秒和 20 秒）。
- **关闭抗锯齿**（#220）：启动器「超分辨率」列表新增「关闭（无抗锯齿）」：去掉《恶魔之魂》自带的 TAA 及其所需的镜头抖动，
  画面更锐利，但边缘有锯齿（`run.cmd -Antialiasing off`）。
- **为插帧和超分留出显存**（#215）：DLSS 插帧、Linux 内置的 FSR 3 插帧，以及 DLSS 或 FSR 3 超分，现在都在显卡的显存预算里
  留有空间（插帧 1.5 GB，超分再加 0.5 GB），与 Windows 上的 FSR 插帧原有的做法相同。一张 RTX 5080（16 GB）在 3440x1440 下
  开 DLSS 和 DLSS 插帧时，以前被当作显存充裕的显卡，游玩后期显存耗尽。`KYTY_FSR_VRAM_RESERVE_MB` 可以设定插帧所留的大小；
  日志会写出预留了多少。
- **纹理缩减时内存更少**：纹理缩减（显存小于 12 GB 的显卡默认开启）不把大纹理的最大一级放进显存；现在游戏本身也不再加载
  这些级别。在波雷塔利亚，游戏加载的纹理从 2.7 GB 降到 1.8 GB，内存少用约 1 GB，画面不变。
- **16 GB 电脑更快归还内存**（#174）：在世界之间往返后，游戏释放的内存归还给 Windows 的速度快了一倍（我们的测试路线上
  2.3 GB，原来 1.1 GB）。在频繁换页的电脑上，归还内存也不再在 Windows 变慢时让游戏每次停住最多一秒。
- **16 GB 电脑上的读回页**（#159、#174）：v20261026 在可用内存少于 2 GB 时，用保护页保护从显卡读回的内存页（蓝屏修复）。
  保护页偶尔会让另一个线程读到旧的页（某个纹理闪一下），而 16 GB 电脑大部分时间可用内存都在 2 GB 左右。现在低于 1.5 GB
  才改用保护页（所有报告的蓝屏都发生在这个值以下），回到 2 GB 后恢复（原为 3 GB）。
- **内存超过 20 GB 的电脑读回更少**（PR #219，@DevList1454）：游戏读取显卡写入的内存时，相邻的缓冲区现在在同一次复制中
  一起读回。在波雷塔利亚 1-3，每帧读回从约 51 次降到 19 次；在 @DevList1454 的测量中，帧率从 50.2 提高到 52.2 fps。
- **回到 Big Picture**（#217）：从 Steam 的 Big Picture 启动时，退出游戏后焦点会交还给 Big Picture，无需点击即可继续用
  手柄操作。
- **Steam Deck**（Linux 版）：Deck 的 GPU 现在原生运行游戏的 64 宽计算着色器（这些着色器的 GPU 时间少 23%；GPU 固定
  1600 MHz 时，1-1 从 20.6 提升到 22.1 fps）；模拟器用于内存跟踪的 GPU 工作在所有显卡上都更轻（Deck 上每帧 7.2 降到
  0.9 ms）。离 30 fps 还有距离。
- **感谢** @DevList1454 的 PR #219，以及 @crni20zd、@Psychowarden、@javsp、@ft1704、@thxnk420、@Myoko 和 @brokeneggg 的反馈与测试。

## v20261026 的变化

已有的预编译着色器继续有效：着色器与 v20261023 相同。预编译只会补上新增着色器提示的管线（#212）。

- **1.005 版本的崩溃**（#185、#204）：在部分电脑上，《恶魔之魂》1.005（欧版和美版）游玩几分钟后、甚至启动时就会崩溃。
  游戏的资源加载器可能取到另一个线程正在撤回的加载请求。模拟器现在让加载器等撤回完成再继续；每次起作用时，日志里会出现
  `Demon's Souls loader guard: ... waited out`。
- **16 GB 电脑蓝屏**（#91）：从 v20261024 起，游戏从显卡读回的内存页所用的保护方式，在内存不足时可能让 Windows 蓝屏
  （MEMORY_MANAGEMENT）。现在只要可用内存少于 2 GB，这些页就改用保护页（guard page），它在我们的高内存压力测试中没有出现问题。
- **着色器准备保存到磁盘**（#200）：后台着色器准备和启动预热现在把翻译结果保存在 `_PipelineCache\static\<游戏>_<版本>.prefetch`
  （1-2 GB）。更新后第一次启动照常翻译；之后的启动约一秒就能读完。我们台式机上的启动预热从 22 秒变为 1 秒，核心少的
  电脑也不用在游戏旁边翻译好几分钟了。`run.cmd -Set KYTY_PREFETCH_CACHE=0` 可以关闭。基于 @Psychowarden 的补丁。
- **受光照的地面和水面闪白**（#196）：后台线程复制缓冲区时，可能取到游戏已经为下一帧改写的数据。现在复制会在游戏得知显卡
  用完这些数据之前完成。靠 @Psychowarden 的测试找到原因。
- **镜头切换时的插帧**（#174）：对话或过场镜头结束时，中间插出的那一帧会把两个镜头的画面混在一起（都半透明）。选择了超分
  （DLSS 或 FSR）时，插帧现在会在游戏的镜头切换处重新开始。
- **1440p 输出使用 4K 界面布局**（#187、#202）：1440p 输出时，界面按 4K 布局、以输出尺寸排版，而不是把 1080p 布局放大：
  文字清晰，角色创建和装备菜单里的角色也在正确位置。`run.cmd -Set KYTY_OUTPUT_LAYOUT=hd` 保留旧方式。
- **1440p 下的宽屏**（#199、#198）：21:9 或更宽的比例配合 1440p 输出时，画面现在按这个高度生成（21:9 时为 3416x1440），
  而不是 16:9（RTX 5080 在 43:18 下从 77 fps 提升到 90 fps）；宽屏输出时过场影片也不再被压扁。基于 @Psychowarden 的补丁。
- **音频输出**（#207）：启动器「游戏」页新增「音频输出」：电视、耳机、AV 功放或条形音箱。游戏会得知你的选择并据此混音，与
  PS5 相同（`run.cmd -Set KYTY_AUDIO_OUTPUT=headphones`）。由 @Psychowarden 发现。
- **用手柄退出游戏**（#203）：按住 PS 键（Xbox 手柄的 Guide 键）两秒即可退出，与 PS5 的「关闭游戏」相同。如果 Steam 或
  Game Bar 占用了这个键，`run.cmd -Set KYTY_PAD_QUIT=both` 让同时按住 Create 和 Options（Xbox 手柄的 View 和 Menu）也能
  退出。Xbox 手柄的 View 键现在等同于按下触摸板（#197）。
- **更多着色器提示**（#212）：一份打到两个结局的完整游玩记录补上了发布包提示中没有的 237 个着色器和 268 条管线：游戏后期
  第一次出现的卡顿更少。感谢 @Vorshim92。
- **预编译**（#213）：某个分片停止工作时不再拖住整个预编译。它 20 分钟没有进展就会重新启动（最多两次），之后跳过；游戏在
  第一次需要时再编译这些管线。
- **16 GB 电脑内存占用更少**：游戏引擎释放的内存归还给 Windows 的速度快了四倍：长时间游玩少占 0.8 GB。
- **新建图像先清零**（PR #193，@intellq）：与 PS5 一样，新图像在第一次使用前为零。以前在 1080p 的 FSR 加 43:18 压缩下，
  第一帧会读到未定义的内容，整局都保持绿色或红色的画面，主要出现在 Linux 的 AMD 显卡上。
- **Linux**：AMD 显卡（RADV）现在会启用传输队列，读回显卡已完成的数据时不再排在整个图形队列后面：RX 9060 XT 和
  RX 9070 XT 行走时帧率提升 19% 到 27%。由 @the2ndday 发现，@dreamerk24m 和 @yazuc 测试。`-Set KYTY_RADV_TRANSFER_QUEUE=0`
  可以关闭；较旧 AMD 显卡的用户如果遇到显卡挂起请反馈。同样来自 @the2ndday：实验性的 `KYTY_UFFD_WP=1`（用 userfaultfd 监视
  GPU 跟踪的写入，需要 Linux 6.4+；PR #206），以及 Linux 上的测试辅助命令（PR #211）。
- **感谢** @Psychowarden、@the2ndday、@Vorshim92 和 @intellq 的补丁与 PR，以及 @danielmazurek79、@Theoryedz、
  @ededaelita1-afk、@ft1704、@dreamerk24m、@Fantomas3228、@yazuc、@marcelofis、@SanyaSan 和 @kvothe3323 的反馈与测试。
