package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

type childProcess struct{ pid, uid int }

func processStatus(pid int) (map[string]string, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return nil, err
	}
	fields := map[string]string{}
	for _, line := range strings.Split(string(data), "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok {
			fields[key] = strings.TrimSpace(value)
		}
	}
	return fields, nil
}

func childProcesses() ([]childProcess, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, err
	}
	var children []childProcess
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || !entry.IsDir() {
			continue
		}
		status, err := processStatus(pid)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		uids := strings.Fields(status["Uid"])
		if len(uids) != 4 {
			return nil, errors.New("process did not expose its Linux UID set")
		}
		uid, err := strconv.Atoi(uids[0])
		if err != nil {
			return nil, err
		}
		if uid >= 20000 && uid <= 29999 {
			children = append(children, childProcess{pid: pid, uid: uid})
		}
	}
	return children, nil
}

func verifySupervisorIdentity() error {
	status, err := processStatus(1)
	if err != nil {
		return err
	}
	if strings.Join(strings.Fields(status["Uid"]), ",") != "0,0,0,0" || status["NoNewPrivs"] != "1" || status["CapEff"] != "00000000000000e1" {
		return errors.New("supervisor must be root with no-new-privileges and only CHOWN/KILL/SETGID/SETUID")
	}
	command, err := os.ReadFile("/proc/1/cmdline")
	if err != nil {
		return err
	}
	if string(command) != "/usr/local/bin/orka-acp-runtime\x00" {
		return errors.New("container did not start the production supervisor entrypoint")
	}
	return nil
}

func verifyProcess(pid, uid int) error {
	status, err := processStatus(pid)
	if err != nil {
		return err
	}
	for _, field := range []string{"Uid", "Gid"} {
		values := strings.Fields(status[field])
		if len(values) != 4 {
			return errors.New("ACP process has no complete Linux identity")
		}
		for _, value := range values {
			if value != strconv.Itoa(uid) {
				return errors.New("ACP child did not retain its private session UID/GID")
			}
		}
	}
	if status["NoNewPrivs"] != "1" || status["CapEff"] != "0000000000000000" {
		return errors.New("ACP child retained capabilities or could gain privileges")
	}
	command, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return err
	}
	if !strings.Contains(string(command), "/opt/agentkit/bin/agentkit-serve\x00--config\x00/agent/agent.yaml\x00--protocol\x00acp\x00") {
		return errors.New("session process is not the actual AgentKit ACP child")
	}
	return nil
}

func verifyPrivateDirectory(root string, uid int) error {
	for _, path := range []string{root, filepath.Join(root, "home"), filepath.Join(root, "workspace")} {
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		stat, ok := info.Sys().(*syscall.Stat_t)
		if !ok || !info.IsDir() || info.Mode().Perm() != 0o700 || stat.Uid != uint32(uid) || stat.Gid != uint32(uid) {
			return errors.New("session tree is not private to the ACP child's UID/GID")
		}
	}
	return nil
}
