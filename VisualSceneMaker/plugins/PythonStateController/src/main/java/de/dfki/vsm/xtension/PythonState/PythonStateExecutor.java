/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package de.dfki.vsm.xtension.PythonState;

import de.dfki.vsm.event.EventDispatcher;
import de.dfki.vsm.event.EventListener;
import de.dfki.vsm.event.EventObject;
import de.dfki.vsm.event.event.VariableChangedEvent;
import de.dfki.vsm.model.project.PluginConfig;
import de.dfki.vsm.runtime.activity.AbstractActivity;
import de.dfki.vsm.runtime.activity.executor.ActivityExecutor;
import de.dfki.vsm.runtime.activity.scheduler.ActivityWorker;
import de.dfki.vsm.runtime.interpreter.value.StringValue;
import de.dfki.vsm.runtime.interpreter.value.BooleanValue;
import de.dfki.vsm.runtime.project.RunTimeProject;
import de.dfki.vsm.util.log.LOGConsoleLogger;
import io.javalin.websocket.WsConnectContext;

import java.io.*;
import java.net.Socket;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

/**
 * PythonStateExecutor launches a Python process, connects to it over TCP,
 * and listens for messages such as EMOTION:happy. Detected emotions
 * are written into a SceneFlow variable (default: python_state).
 */
public class PythonStateExecutor extends ActivityExecutor implements EventListener {

    static long sUtteranceId = 0;
    // Track active threads in the current scene
    private final Map<String, ActivityWorker> mActivityWorkerMap = new HashMap<>();

    private static final String FEATURE_PYTHON_BIN          = "pythonBin";
    private static final String FEATURE_SCRIPT_PATH         = "scriptPath";
    private static final String FEATURE_HOST                = "127.0.0.1";
    private static final String FEATURE_PORT                = "5002";
    private static final String FEATURE_PYTHON_CONNECTED_VAR= "python_connected_var";
    private static final String FEATURE_SCENEFLOW_STATE_VAR = "sceneflowStateVar";

    // defaults
    private static final String DEFAULT_PYTHON_BIN            = "python3";
    private static final String DEFAULT_HOST                  = "127.0.0.1";
    private static final String DEFAULT_PORT                  = "5002";
    private static final String DEFAULT_PYTHON_CONNECTED_VAR  = "python_connected";
    private static final String DEFAULT_SCENEFLOW_STATE_VAR   = "python_state";

    private Process pyProcess;
    private ScheduledExecutorService scheduler;
    private String pythonBin;
    private String scriptPath;
    private String pythonConnectedVar;
    private String host;
    private int port;
    private String sceneVar;

    private final LOGConsoleLogger mLogger = LOGConsoleLogger.getInstance();
    private final ArrayList<WsConnectContext> mWebsockets = new ArrayList<>();

    public PythonStateExecutor(final PluginConfig config,
                               final RunTimeProject project) {
        super(config, project);
    }

    @Override
    public synchronized String marker(long id) {
        return "$(" + id + ")";
    }

    @Override
    public void launch() {
        EventDispatcher.getInstance().register(this);

        mLogger.message(">>> PythonStateExecutor.launch() called");
        System.out.println("Loading python message sender and receiver ...");

        pythonConnectedVar = mConfig.getProperty(FEATURE_PYTHON_CONNECTED_VAR,
                DEFAULT_PYTHON_CONNECTED_VAR);
        pythonBin          = mConfig.getProperty(FEATURE_PYTHON_BIN,
                DEFAULT_PYTHON_BIN);
        scriptPath         = mConfig.getProperty(FEATURE_SCRIPT_PATH);
        host               = mConfig.getProperty(FEATURE_HOST, DEFAULT_HOST);
        port               = Integer.parseInt(
                mConfig.getProperty(FEATURE_PORT, DEFAULT_PORT)
        );
        sceneVar           = mConfig.getProperty(FEATURE_SCENEFLOW_STATE_VAR,
                DEFAULT_SCENEFLOW_STATE_VAR);

        // 1) Launch the Python process
        try {
            File workDir = new File(scriptPath).getParentFile();
            pyProcess = new ProcessBuilder(pythonBin, scriptPath)
                    .directory(workDir)
                    .inheritIO()
                    .start();
        } catch (IOException e) {
            System.out.println("Failed to start Python: " + e.getMessage());
            return;
        }

        System.out.println("Python Started");

        // 2) Connect to Python server and listen for messages
        new Thread(() -> {
            try {
                Socket socket = null;
                // retry until Python server is up
                while (socket == null) {
                    try {
                        socket = new Socket(host, port);
                    } catch (IOException retry) {
                        try { Thread.sleep(200); } catch (InterruptedException ignored) {}
                    }
                }

                System.out.println(" Connected to Python server");

                // Set connected flag in SceneFlow
                mProject.setVariable(
                        pythonConnectedVar,
                        new BooleanValue(true)
                );

                BufferedReader reader = new BufferedReader(
                        new InputStreamReader(socket.getInputStream())
                );
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println(" From Python: " + line);

                    if (line.startsWith("INTERVENTION:")) {
                        String emotion = line.split(":")[1].trim();
                        mProject.setVariable(sceneVar, new StringValue(emotion));
                        System.out.println(" Emotion updated in SceneFlow: " + emotion);
                    }
                }
                socket.close();
            } catch (IOException e) {
                System.out.println("️ Could not connect/read from Python: " + e.getMessage());
            }
        }).start();
    }

    @Override
    public void unload() {
        if (scheduler != null) scheduler.shutdownNow();
        if (pyProcess  != null)    pyProcess.destroy();
        mWebsockets.clear();
        EventDispatcher.getInstance().remove(this);
    }

    @Override
    public void execute(AbstractActivity activity) {
        final String activity_actor = activity.getActor();
    }

    @Override
    public void update(EventObject event) {
        if (event instanceof VariableChangedEvent) {
            VariableChangedEvent ev = (VariableChangedEvent) event;
            String info = event.toString();
            info = info.split("\\(")[1].split(",")[0] + "#" + info.split("#c#")[1].split("\\)")[0];
            String varListUpdatesMessage = "VSMMessage#UPDATE#" + info.replace("'", "");

            try {
                mWebsockets.forEach(websocks -> websocks.send(varListUpdatesMessage));
            } catch (Exception e) {
                mLogger.failure(e.toString());
            }
        }
    }
}
